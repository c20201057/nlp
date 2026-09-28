"""Task 2：100 维词向量取平均作为文档表示，分类器为 Logistic Regression。

三组实验：预训练 GloVe 词向量 glove.6B.100d、在 AG News 上训练的 Word2Vec、在 NYT 上训练的 Word2Vec。
"""
import argparse
import os
import time

import numpy as np

from common import (CACHE_DIR, ROOT, SEED, fit_lr_with_val, load_ag_texts,
                    load_nyt_splits, save_result, tokenize_all)

DIM = 100
GLOVE_PATH = os.path.join(ROOT, "embeddings", "glove.6B.100d.txt")
W2V_PARAMS = dict(vector_size=DIM, window=5, min_count=2, sg=1, negative=5, epochs=10,
                  seed=SEED, workers=1)  # 单线程训练，保证结果可复现


def load_glove(path):
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, *vals = line.rstrip().split(" ")
            vectors[word] = np.asarray(vals, dtype=np.float32)
    return vectors


def train_word2vec(corpus, name):
    from gensim.models import Word2Vec

    path = os.path.join(CACHE_DIR, f"w2v_{name}.model")
    if os.path.exists(path):
        return Word2Vec.load(path).wv
    t0 = time.time()
    model = Word2Vec(sentences=corpus, **W2V_PARAMS)
    print(f"  trained word2vec on {name}: {len(corpus)} docs, "
          f"|V|={len(model.wv)}, {time.time() - t0:.0f}s")
    os.makedirs(CACHE_DIR, exist_ok=True)
    model.save(path)
    return model.wv


def doc_vectors(docs, lookup):
    """e(d) = (1/n) Σ e(w_i)，只对词向量表中存在的有效单词取平均；没有有效单词的文档为零向量。"""
    X = np.zeros((len(docs), DIM), dtype=np.float32)
    covered = 0
    for i, doc in enumerate(docs):
        vecs = [lookup[w] for w in doc if w in lookup]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
            covered += 1
    return X, covered


def run(name, lookup, tokens, y, labels, extra):
    print(f"== {name}")
    X, stats = {}, {}
    for split in ("train", "val", "test"):
        X[split], covered = doc_vectors(tokens[split], lookup)
        n_tok = sum(len(d) for d in tokens[split])
        n_hit = sum(w in lookup for d in tokens[split] for w in d)
        stats[split] = {"docs_with_vectors": covered, "docs": len(tokens[split]),
                        "token_coverage": round(n_hit / n_tok, 4)}
    print(f"  token coverage (test) = {stats['test']['token_coverage']}")

    result = fit_lr_with_val(X["train"], y["train"], X["val"], y["val"], X["test"], y["test"],
                             labels, scaler=True)
    result.update(extra, coverage=stats)
    save_result(name, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["glove", "w2v_ag", "w2v_nyt"])
    args = parser.parse_args()

    splits, labels = load_nyt_splits()
    tokens = {name: tokenize_all(texts, f"nyt_{name}") for name, (texts, _) in splits.items()}
    y = {name: labels_ for name, (_, labels_) in splits.items()}

    if args.only in (None, "glove"):
        glove = load_glove(GLOVE_PATH)
        run("task2_glove", glove, tokens, y, labels,
            {"embedding": "glove.6B.100d", "embedding_vocab": len(glove)})
        del glove

    if args.only in (None, "w2v_ag"):
        ag_tokens = tokenize_all(load_ag_texts(), "ag")
        wv = train_word2vec(ag_tokens, "ag")
        run("task2_w2v_ag", wv, tokens, y, labels,
            {"embedding": "word2vec trained on AG News", "w2v_params": W2V_PARAMS,
             "embedding_vocab": len(wv), "train_docs": len(ag_tokens)})

    if args.only in (None, "w2v_nyt"):
        # 只用 NYT 训练集训练词向量，避免 val / test 文本参与表示学习
        wv = train_word2vec(tokens["train"], "nyt_train")
        run("task2_w2v_nyt", wv, tokens, y, labels,
            {"embedding": "word2vec trained on NYT training split", "w2v_params": W2V_PARAMS,
             "embedding_vocab": len(wv), "train_docs": len(tokens["train"])})


if __name__ == "__main__":
    main()
