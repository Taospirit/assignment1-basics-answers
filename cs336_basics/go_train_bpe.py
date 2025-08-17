import json
import time
import sys
import os
import regex as re
import numpy as np

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from cs336_basics.impl_bpe_tokenizer import BPE_Tokenizer
from cs336_basics.impl_train_bpe import PAT, train_bpe

def save_bpe_vocab_merges(vocab, merges, output_path, overwrite=False):
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    
    # Save vocab in GPT-2 format (token string -> token ID)
    vocab_gpt2 = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to GPT-2 unicode representation
        token_str = "".join(gpt2_byte_encoder[b] for b in token_bytes)
        vocab_gpt2[token_str] = token_id
    
    vocab_save_path = output_path + "-vocab.json"
    if not os.path.exists(vocab_save_path) or overwrite:
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            json.dump(vocab_gpt2, f, indent=4)
    else:
        print(f"vocab file {vocab_save_path} already exists, skip saving!")
    
    merges_save_path = output_path + "-merges.txt"
    if not os.path.exists(merges_save_path) or overwrite:
        # Save merges in GPT-2 format (unicode string pairs)
        with open(merges_save_path, "w", encoding="utf-8") as f:
            for merge in merges:
                # Convert each byte in the merge to GPT-2 unicode representation
                merge_str1 = "".join(gpt2_byte_encoder[b] for b in merge[0])
                merge_str2 = "".join(gpt2_byte_encoder[b] for b in merge[1])
                f.write(f"{merge_str1} {merge_str2}\n")

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return BPE_Tokenizer(vocab, merges, special_tokens)

def get_chunks_from_dataset(dataset_name):
    root_dir = "/home/lintao/llm_codes/cs336"
    input_path = root_dir + f"/data/{dataset_name}.txt"
    assert os.path.exists(input_path), f"input path {input_path} does not exist!"
    special_tokens = ["<|endoftext|>"]
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    return chunks

def get_words_from_chunks(chunks):
    words = []
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            words.append(word)
    return words

def train_bpe_dataset(
    dataset_name, 
    vocab_size, 
    special_tokens=["<|endoftext|>"],
    output_path=None,
    overwrite=False,
):
    root_dir = "/home/lintao/llm_codes/cs336"
    input_path = root_dir + f"/data/{dataset_name}.txt"
    assert os.path.exists(input_path), f"input path {input_path} does not exist!"
    output_path = root_dir + f"/data/{dataset_name}-bpe-{vocab_size}"
    
    begin_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    end_time = time.time()
    print(f"vocab size: {len(vocab)}, \nmerges size: {len(merges)}, \ndataset_name: {dataset_name}, \ntime: {end_time - begin_time}")

    save_bpe_vocab_merges(vocab, merges, output_path, overwrite)
    
if __name__ == "__main__":
    # 先处理较小的数据集
    import random
    random.seed(42)

    vocab_size = 32000
    dataset_name = "TinyStoriesV2-GPT4-valid"
    print(f"Processing {dataset_name}...")
    train_bpe_dataset(dataset_name, vocab_size)

    chunks_str = get_chunks_from_dataset(dataset_name)
    print(f"text length: {len(chunks_str)}")
    
    vocab_path = f"/home/lintao/llm_codes/cs336/data/{dataset_name}-bpe-{vocab_size}-vocab.json"
    merges_path = f"/home/lintao/llm_codes/cs336/data/{dataset_name}-bpe-{vocab_size}-merges.txt"
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)

    compression_ratio = []
    for text_str in random.sample(chunks_str, 100):
        encode_ids = tokenizer.encode(text_str)
        decode_text = tokenizer.decode(encode_ids)
        assert text_str == decode_text

        bytes_len = len(text_str.encode('utf-8'))
        bpe_len = len(encode_ids)
        compression_ratio.append(bytes_len / bpe_len)

    cratio = np.array(compression_ratio)
    print(f"compression_ratio|size: {len(cratio)}, mean: {np.mean(cratio)}, min: {np.min(cratio)}, max: {np.max(cratio)}")