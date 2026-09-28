"""Task 1：Binary Bag of Words 与 Word Frequency，分类器为 Logistic Regression。"""
from sklearn.feature_extraction.text import CountVectorizer

from common import fit_lr_with_val, load_nyt_splits, save_result, tokenize_all


def identity(tokens):
    return tokens


def main():
    splits, labels = load_nyt_splits()
    tokens = {name: tokenize_all(texts, f"nyt_{name}") for name, (texts, _) in splits.items()}
    y = {name: labels_ for name, (_, labels_) in splits.items()}

    for name, binary in [("task1_binary_bow", True), ("task1_word_frequency", False)]:
        print(f"== {name}")
        # 词表只由训练集构建，|V| 维稀疏向量；binary=True 记录是否出现，False 记录出现次数
        vec = CountVectorizer(analyzer=identity, binary=binary)
        X_train = vec.fit_transform(tokens["train"])
        X_val, X_test = vec.transform(tokens["val"]), vec.transform(tokens["test"])
        print(f"  |V| = {len(vec.vocabulary_)}")

        result = fit_lr_with_val(X_train, y["train"], X_val, y["val"], X_test, y["test"], labels)
        result["vocab_size"] = len(vec.vocabulary_)
        save_result(name, result)


if __name__ == "__main__":
    main()
