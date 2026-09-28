# NLP 作业一：文本分类实现

本仓库在 NYT 新闻数据集上比较三类文本表示方法。NYT 数据集共有 business、politics、sports 三个类别。

- **Task 1**：Binary Bag of Words 与 Word Frequency 两种词袋表示，分类器为 Logistic Regression。
- **Task 2**：用 100 维词向量的平均值表示文档，分类器为 Logistic Regression。词向量分别来自预训练的 GloVe 6B、在 AG News 上训练的 Word2Vec、在 NYT 上训练的 Word2Vec。
- **Task 3**：微调 `google-bert/bert-base-uncased`，max_length 为 64，训练 3 个 epoch。

全部实验在一张 NVIDIA A40 上完成，完整运行输出见 [`logs/run_all.log`](logs/run_all.log)。

## 实验结果

下表为 NYT Test Set 上的结果，测试集共 1153 条。

| Task | 方法 | Accuracy | Macro-F1 |
|---|---|---|---|
| 1 | Binary Bag of Words + LR | 0.9870 | 0.9712 |
| 1 | Word Frequency + LR | 0.9879 | 0.9724 |
| 2 | GloVe 6B-100d + LR | **0.9922** | **0.9826** |
| 2 | AG News 训练的 Word2Vec + LR | 0.9844 | 0.9653 |
| 2 | NYT 训练的 Word2Vec + LR | 0.9879 | 0.9725 |
| 3 | BERT-base-uncased 微调 | 0.9792 | 0.9530 |

每组实验的逐类别 Precision、Recall、F1，验证集结果，以及超参数搜索过程都保存在 `results/` 下对应的 JSON 文件中。

## 仓库结构

```
├── HW-1/
│   ├── nyt.csv               课程提供的 NYT 数据，含 text 和 label 两列，共 11519 条
│   └── ag.csv                课程提供的 AG News 数据，只有 text 一列，共 90000 条，仅用于训练 Word2Vec
├── src/
│   ├── common.py             三个 Task 共用的数据划分、分词、评价指标、LR 训练与结果保存
│   ├── task1_bow.py          Task 1
│   ├── task2_embeddings.py   Task 2，可用 --only glove、--only w2v_ag 或 --only w2v_nyt 只跑其中一组
│   └── task3_bert.py         Task 3
├── scripts/
│   ├── setup_env.sh          创建独立的 Python 环境并下载 BERT 权重
│   ├── download_glove.sh     下载 glove.6B.100d.txt
│   └── run_all.sh            依次运行三个 Task
├── nltk_data/                word_tokenize 所需的英文 punkt_tab 分词模型
├── results/                  各实验的结果 JSON，以及记录数据划分下标的 nyt_split.json
├── logs/                     运行日志
└── requirements.txt
```

`env/`、`models/`、`embeddings/`、`cache/` 四个目录体积较大，没有放进仓库，运行脚本时会自动生成或下载。它们分别存放 Python 环境、BERT 权重、GloVe 词向量，以及分词结果和 Word2Vec 模型的缓存。

## 运行方法

需要一台带 NVIDIA GPU 的 Linux 机器，并事先装好 micromamba 或 conda。以下命令都在仓库根目录下执行。

```bash
git clone -b HW1 https://github.com/c20201057/nlp.git && cd nlp
bash scripts/setup_env.sh        # 创建 env/ 并下载 BERT 权重到 models/
bash scripts/download_glove.sh   # 下载 GloVe 到 embeddings/
bash scripts/run_all.sh          # 依次运行 Task 1、2、3
```

`setup_env.sh` 会创建 Python 3.10 环境，安装 torch 2.4.1+cu118 和 `requirements.txt` 中的依赖。如果无法直连 huggingface.co，先执行 `export HF_ENDPOINT=https://hf-mirror.com` 再运行它。

`run_all.sh` 会覆盖 `results/` 中已有的结果，并把完整输出写入 `logs/run_all.log`。在 A40 上完整运行一遍约需 15 分钟，其中 BERT 训练只占 1 分钟左右，其余时间主要花在 Logistic Regression 的超参数搜索和 Word2Vec 训练上。

也可以单独运行某一个 Task：

```bash
export PYTHONNOUSERSITE=1
env/bin/python src/task1_bow.py
env/bin/python src/task2_embeddings.py --only glove
env/bin/python src/task3_bert.py --model models/bert-base-uncased
```

## 实验设置

**数据划分。** 用随机种子 42 打乱 NYT 后，按 80%、10%、10% 的比例划分为训练集 9215 条、验证集 1151 条、测试集 1153 条。所有 Task 共用同一份划分，各部分的下标记录在 `results/nyt_split.json` 中。

**分词。** 统一使用 `nltk.word_tokenize`。数据中的文本已经是小写。

**Logistic Regression。** 使用 scikit-learn 的 `LogisticRegression`，求解器为 lbfgs，max_iter 为 5000。正则强度 C 在 0.01、0.1、1、10、100 中选择，以验证集上的 Macro-F1 为准，选定后只在测试集上评测一次。

**Task 1 词表。** 词表只由训练集构建，共 128343 个词，文档向量用稀疏矩阵存储。

**Task 2 文档向量。** 对文档中所有出现在词向量词表里的单词取词向量平均值。送入 Logistic Regression 之前先做标准化，均值和方差只在训练集上估计。

**Word2Vec。** 使用 gensim 4.3.3 的 skip-gram 模型，vector_size 为 100，window 为 5，min_count 为 2，negative 为 5，训练 10 个 epoch，seed 为 42。训练只用单线程，这样多次运行得到的词向量完全相同。NYT 版本的词向量只用训练集文本训练，避免验证集和测试集的文本参与表示学习。

**BERT。** max_length 为 64，训练 3 个 epoch，batch size 为 32。优化器为 AdamW，学习率 2e-5，weight decay 0.01，前 10% 的步数线性 warmup，之后线性衰减。训练使用 bf16 混合精度。报告的是第 3 个 epoch 结束时的模型。

**硬件与软件。** 一张 NVIDIA A40 显卡，显存 46GB。Python 3.10，PyTorch 2.4.1+cu118，transformers 4.46.3，scikit-learn 1.5.2，gensim 4.3.3，nltk 3.9.1。
