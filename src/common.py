"""三个 Task 共用的数据划分、分词、评价与结果保存。"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("HW1_DATA_DIR", os.path.join(ROOT, "HW-1"))
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR = os.path.join(ROOT, "cache")
SEED = 42


def load_nyt_splits():
    """随机打乱 NYT 后按 80% / 10% / 10% 划分为 train / val / test。

    所有 Task 都调用这里，保证使用同一份划分；划分下标写入 results/nyt_split.json 以便核对。
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "nyt.csv"))
    labels = sorted(df["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}

    perm = np.random.RandomState(SEED).permutation(len(df))
    n_train, n_val = int(0.8 * len(df)), int(0.1 * len(df))
    idx = {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "nyt_split.json"), "w") as f:
        json.dump({"seed": SEED, "labels": labels,
                   **{k: v.tolist() for k, v in idx.items()}}, f)

    splits = {}
    for name, ids in idx.items():
        part = df.iloc[ids]
        splits[name] = (part["text"].astype(str).tolist(),
                        part["label"].map(label2id).to_numpy())
    return splits, labels


def load_ag_texts():
    return pd.read_csv(os.path.join(DATA_DIR, "ag.csv"))["text"].astype(str).tolist()


def tokenize_all(texts, cache_name):
    """nltk.word_tokenize 分词；结果缓存到 cache/，避免每个实验重复分词。"""
    path = os.path.join(CACHE_DIR, cache_name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(texts):
            return cached

    import nltk
    from nltk import word_tokenize
    # word_tokenize 依赖的 punkt_tab 模型随仓库放在 nltk_data/ 下
    nltk.data.path.insert(0, os.path.join(ROOT, "nltk_data"))

    t0 = time.time()
    tokens = [word_tokenize(t.lower()) for t in texts]
    print(f"[tokenize] {cache_name}: {len(texts)} docs in {time.time() - t0:.1f}s")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokens, f)
    return tokens


def evaluate(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, target_names=labels,
                                        digits=4, output_dict=True),
    }


def fit_lr_with_val(X_train, y_train, X_val, y_val, X_test, y_test, labels,
                    C_grid=(0.01, 0.1, 1.0, 10.0, 100.0), scaler=False):
    """Logistic Regression：在 validation set 上按 Macro-F1 选正则强度 C，再在 test set 上报告。

    scaler=True 时先对稠密的文档向量做标准化，均值和方差只在训练集上拟合。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    search = []
    best = None
    for C in C_grid:
        t0 = time.time()
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        if scaler:
            clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        val = evaluate(y_val, clf.predict(X_val), labels)
        search.append({"C": C, "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                       "fit_seconds": round(time.time() - t0, 1)})
        print(f"  C={C:<6} val acc={val['accuracy']:.4f} macro_f1={val['macro_f1']:.4f}")
        if best is None or val["macro_f1"] > best[1]["macro_f1"]:
            best = (clf, val, C)

    clf, val, C = best
    return {
        "best_C": C,
        "c_search": search,
        "val": val,
        "test": evaluate(y_test, clf.predict(X_test), labels),
    }


def save_result(name, result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    test = result["test"]
    print(f"[{name}] test accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}")
