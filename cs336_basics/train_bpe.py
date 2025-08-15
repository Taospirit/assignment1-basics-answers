import json
import time
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

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
    output_path = root_dir + f"/data/{dataset_name}-bpe"
    
    begin_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    # vocab, merges = [], []
    end_time = time.time()
    print(f"vocab size: {len(vocab)}, \nmerges size: {len(merges)}, \ndataset_name: {dataset_name}, \ntime: {end_time - begin_time}")
    # Convert to GPT-2 format for saving
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

if __name__ == "__main__":
    # 先处理较小的数据集
    print("Processing TinyStoriesV2-GPT4-valid...")
    train_bpe_dataset("TinyStoriesV2-GPT4-valid", 10000)
    
    # print("Processing TinyStoriesV2-GPT4-train...")
    # train_bpe_dataset("TinyStoriesV2-GPT4-train", 10000)
    
    # 处理较大的数据集（如果内存允许）
    # try:
    #     print("Processing owt_valid...")
    #     train_bpe_dataset("owt_valid", 32000)
    # except Exception as e:
    #     print(f"Error processing owt_valid: {e}")
    
    # try:
    #     print("Processing owt_train...")
    #     train_bpe_dataset("owt_train", 32000)
    # except Exception as e:
    #     print(f"Error processing owt_train: {e}")

    # train_bpe_dataset("owt_valid", 32000)