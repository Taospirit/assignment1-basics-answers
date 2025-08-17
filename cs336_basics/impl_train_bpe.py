import os
import time
import regex as re
import multiprocessing as mp
from collections import defaultdict
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def word_to_bytes_tuple(word: str):
    return tuple(bytes([x]) for x in word.encode("utf-8"))

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(chunk_data):
    """Process a chunk of text for BPE training."""    
    chunk_text, special_tokens = chunk_data    
    pat_re = re.compile(PAT)
    
    pre_token_cnt = defaultdict(int)
    chunks = re.split("|".join(map(re.escape, special_tokens)), chunk_text)
    for chunk in chunks:
        for m in pat_re.finditer(chunk):
            word = m.group(0)
            pre_token_cnt[word_to_bytes_tuple(word)] += 1
    return pre_token_cnt

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
        """
    # 1. Initialize vocabulary with single bytes
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_id = 256

    # 2. Add special tokens
    special_token_bytes = [token.encode('utf-8') for token in special_tokens]
    for token_bytes in special_token_bytes:
        vocab[vocab_id] = token_bytes
        vocab_id += 1

    # 3. Process text using chunked file reading for memory efficiency
    begin_time = time.time()
    
    # 使用分块处理来避免内存错误
    num_workers = mp.cpu_count()
    # 获取文件分块边界
    chunk_data = []
    with open(input_path, 'rb') as f:
        # 使用第一个特殊令牌作为分割点
        split_token = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_data.append((chunk_text, special_tokens))

    # 使用多进程处理分块
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunk_data)

    # 合并结果
    pre_tokens_cnt = defaultdict(int)
    for result in results:
        for k, v in result.items():
            pre_tokens_cnt[k] += v
    print(f"process text time cost: {time.time() - begin_time}, pre-token size {len(pre_tokens_cnt)}")

    begin_time = time.time()
    # 4. Merge tokens (BPE loop)

    pair_counts = defaultdict(int)
    for token_tuple, cnt in pre_tokens_cnt.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += cnt
    if not pair_counts:
        return vocab, []

    def update_pair_counts(token, idx, cnt):
        new_token = token[idx] + token[idx + 1]

        if idx > 0:
            left_pair = (token[idx - 1], token[idx])
            pair_counts[left_pair] -= cnt
            if pair_counts[left_pair] <= 0:
                del pair_counts[left_pair]
            left_new_pair = (left_pair[0], new_token)
            pair_counts[left_new_pair] += cnt

        if idx < len(token) - 2:
            right_pair = (token[idx + 1], token[idx + 2])
            pair_counts[right_pair] -= cnt
            if pair_counts[right_pair] <= 0:
                del pair_counts[right_pair]
            right_new_pair = (new_token, right_pair[1]) 
            pair_counts[right_new_pair] += cnt

    merges = []
    time_cnt = 1
    time_interval = 5.0
    while len(vocab) < vocab_size:
        during = time.time() - begin_time
        if (during > time_cnt * time_interval):
            time_cnt += 1
            print(f"merge tokens time cost: {during}, vocab size: {len(vocab)}")
        if not pair_counts:
            break

        max_count = max(pair_counts.values())
        max_candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(max_candidates)
        del pair_counts[best_pair]

        a, b = best_pair
        new_token = a + b
        vocab[vocab_id] = new_token
        vocab_id += 1

        changes = []
        for token_tuple, cnt in list(pre_tokens_cnt.items()):
            if best_pair[0] not in token_tuple or best_pair[1] not in token_tuple:
                continue

            new_token_tuple = []
            idx = 0
            while idx < len(token_tuple):
                if idx < len(token_tuple) - 1 and token_tuple[idx:idx + 2] == best_pair:
                    # 获得新的pair计数
                    update_pair_counts(token_tuple, idx, cnt)
                    new_token_tuple.append(new_token)
                    idx += 2
                else:
                    new_token_tuple.append(token_tuple[idx])
                    idx += 1

            if len(new_token_tuple) < len(token_tuple):
                changes.append((token_tuple, tuple(new_token_tuple), cnt))

        for old_tuple, new_tuple, cnt in changes:
            # 更新token计数
            pre_tokens_cnt[new_tuple] += cnt
            del pre_tokens_cnt[old_tuple]

        merges.append(best_pair)

    return vocab, merges