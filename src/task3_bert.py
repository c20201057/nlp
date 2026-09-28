"""Task 3：微调 google-bert/bert-base-uncased 做 NYT 文本分类，max_length=64，训练 3 个 epoch。"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import SEED, evaluate, load_nyt_splits, save_result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, labels, max_length):
    enc = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length",
                    return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    preds = []
    for input_ids, attention_mask, _ in loader:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=attention_mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased",
                        help="模型名或本地目录")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=0, help="仅用于冒烟测试，>0 时提前停止")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'} "
          f"amp={amp_dtype}")

    splits, labels = load_nyt_splits()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {name: encode(tokenizer, texts, y, args.max_length)
            for name, (texts, y) in splits.items()}
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, generator=g)
    eval_loaders = {name: DataLoader(data[name], batch_size=128) for name in ("val", "test")}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps)

    history, step = [], 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for input_ids, attention_mask, y in train_loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                             labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        n_batches = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
        val = evaluate(splits["val"][1], predict(model, eval_loaders["val"], device, amp_dtype), labels)
        history.append({"epoch": epoch, "train_loss": running / n_batches,
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: loss={running / n_batches:.4f} val acc={val['accuracy']:.4f} "
              f"macro_f1={val['macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        if args.max_steps and step >= args.max_steps:
            break

    # 按作业要求训练满 3 个 epoch，报告最后一个 epoch 的模型在 test set 上的结果
    test = evaluate(splits["test"][1], predict(model, eval_loaders["test"], device, amp_dtype), labels)
    result = {
        "model": args.model,
        "config": {k: v for k, v in vars(args).items() if k != "model"},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp_dtype": str(amp_dtype),
        "train_seconds": round(time.time() - t_start, 1),
        "history": history,
        "val": val,
        "test": test,
    }
    save_result("task3_bert" if not args.max_steps else "task3_bert_smoke", result)


if __name__ == "__main__":
    main()
