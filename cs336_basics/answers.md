

## 2.7 
### a
对 tinystories 而言，使用 10k 和 32k 的 vocab_size，最后的压缩比差别不大，均值都是 4 左右。具体数据如下：

Processing TinyStoriesV2-GPT4-valid...
vocab size: 10000, 
merges size: 9743, 
dataset_name: TinyStoriesV2-GPT4-valid, 
time: 15.552928924560547
text length: 27631
compression_ratio|size: 100, mean: 4.052849956003634, min: 3.576419213973799, max: 4.6091370558375635

Processing TinyStoriesV2-GPT4-valid...
vocab size: 18053, 
merges size: 17796, 
dataset_name: TinyStoriesV2-GPT4-valid, 
time: 26.097894191741943
text length: 27631
compression_ratio|size: 100, mean: 4.064364633092085, min: 3.5921052631578947, max: 4.778947368421052

Processing TinyStoriesV2-GPT4-train...
vocab size: 10000, 
merges size: 9743, 
dataset_name: TinyStoriesV2-GPT4-train, 
time: 139.99865865707397
text length: 2717700
compression_ratio|size: 100, mean: 4.089606206462387, min: 2.0, max: 4.461538461538462

Processing TinyStoriesV2-GPT4-train...
vocab size: 32000, 
merges size: 31743, 
dataset_name: TinyStoriesV2-GPT4-train, 
time: 310.8837614059448
text length: 2717700
compression_ratio|size: 100, mean: 4.0992144850703705, min: 2.0, max: 4.578947368421052

### b

### c
### d
为什么选择 uint16？
1. token ID 的范围适配

现代的 BPE 分词器（如 GPT-2 的 tiktoken）通常使用约 50,000 到 32,000 的词汇表大小。uint16 类型可以表示 0 到 65,535 的整数，完全覆盖这些 token ID 的范围，确保不会发生溢出。

2. 内存和存储效率

与默认的 Python 整数类型（通常为 32 位或 64 位）相比，uint16 类型每个 token ID 仅占用 2 字节。对于大规模数据集（如 TinyStories 或 OpenWebText），这可以显著减少内存和磁盘空间的占用。

3. 与 NumPy 和训练框架的兼容性

uint16 是 NumPy 中高效且广泛支持的数据类型，适用于存储和处理大规模的 token ID 序列。在训练语言模型时，使用 uint16 类型的数组可以提高数据加载和处理的效率。