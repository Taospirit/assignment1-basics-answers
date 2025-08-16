from typing import Iterable, Iterator
import os
import regex as re
import json
from typing import Iterable, Iterator
from cs336_basics.imp_train_bpe import PAT, word_to_bytes_tuple

class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.vocab_reverse = {token: token_id for token_id, token in vocab.items()}
        self.special_tokens = list(set(special_tokens)) if special_tokens else []
        
        self.special_token2ids = {}
        next_id = len(self.vocab)
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            special_token_id = self.vocab_reverse.get(token_bytes, next_id)
            self.special_token2ids[token] = special_token_id
            next_id += 1
        self.special_id2tokens = {v: k for k, v in self.special_token2ids.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens. This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None

        def load_file(filepath):
            file_extension = os.path.splitext(filepath)[1]
            if file_extension == '.txt':
                with open(filepath, 'r') as f:
                    return f.read()
            elif file_extension == '.json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # 加载词汇表和合并规则
        vocab = load_file(vocab_filepath)
        merges = load_file(merges_filepath)
        
        # 创建并返回tokenizer实例
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        
        # 处理特殊token
        if self.special_tokens:
            # 使用正则表达式进行更精确的特殊token处理
            # 按长度降序排列特殊token，优先匹配最长的
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            
            # 分割文本，保留分隔符
            chunks = re.split(f'({special_tokens_pattern})', text)
            result = []
            for chunk in chunks:
                if chunk in self.special_token2ids:
                    # 这是一个特殊token
                    result.append(self.special_token2ids[chunk])
                elif chunk:  # 非空的普通文本部分
                    result.extend(self._encode_text_part(chunk, PAT))
            # print(f"result is {result}")
            return result
        
        # 没有特殊token，正常编码
        return self._encode_text_part(text, PAT)

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        if not self.merges:
            return tokens
        
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            # 找所有可合并对及其优先级
            candidate_pairs = [(pair, self.merges.index(pair)) for pair in pairs if pair in self.merges]
            if not candidate_pairs:
                break
            # 找优先级最高的 pair (优先级最低的数字)
            best_pair = min(candidate_pairs, key=lambda x: x[1])[0]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 如果匹配最佳 pair，则合并
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tuple(tokens)
    
    def _encode_text_part(self, text: str, PAT: str) -> list[int]:
        """编码文本的一部分（不包含特殊token）"""
        tokens_list = []
        for m in re.finditer(PAT, text):
            word = m.group(0)
            tokens_list.append(word_to_bytes_tuple(word))

        tokens_id = []
        for tokens in tokens_list:
            merged_tokens = self._apply_merges(tokens)
            tokens_id += [self.vocab_reverse[t] for t in merged_tokens]
        return tokens_id

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        # This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        if len(ids) == 0:
            return ""

        # 分段处理：将普通token和特殊token分开处理
        result_parts = []
        current_bytes = []
        
        for token_id in ids:
            if token_id in self.vocab:
                # 普通token，添加到当前字节序列
                current_bytes.append(self.vocab[token_id])
            elif token_id in self.special_id2tokens:
                # 特殊token：先解码当前字节序列，然后添加特殊token
                if current_bytes:
                    result_bytes = b''.join(current_bytes)
                    result_parts.append(result_bytes.decode('utf-8', errors='replace'))
                    current_bytes = []
                result_parts.append(self.special_id2tokens[token_id])
            else:
                raise ValueError(f"Invalid token ID: {token_id}")
        
        # 处理剩余的字节
        if current_bytes:
            result_bytes = b''.join(current_bytes)
            result_parts.append(result_bytes.decode('utf-8', errors='replace'))
        
        return ''.join(result_parts)

if __name__ == "__main__":
    text = "the cat ate"
    vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    tokenizer = BPE_Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    encoded_ids = tokenizer.encode(text)
    print(encoded_ids, type(encoded_ids))
    decode_str = tokenizer.decode(encoded_ids)
    print(decode_str, type(decode_str))
    assert decode_str == text
