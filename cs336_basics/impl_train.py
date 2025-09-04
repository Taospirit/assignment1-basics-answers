import numpy as np
import torch
import os
from typing import BinaryIO, IO


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """从数据集中随机采样语言建模的输入序列和对应的标签。

    Args:
        dataset: 包含整数token ID的一维numpy数组
        batch_size: 要采样的批次大小
        context_length: 每个采样样本的上下文长度
        device: PyTorch设备字符串（如'cpu'或'cuda:0'）

    Returns:
        包含两个torch.LongTensor的元组，形状为(batch_size, context_length)：
        - 第一个是采样的输入序列
        - 第二个是对应的语言建模标签
    """
    # 计算有效的起始位置范围
    max_start = len(dataset) - context_length
    # 随机选择起始位置
    starts = np.random.randint(0, max_start, size=batch_size)

    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    targets = np.stack([dataset[start + 1 : start + 1 + context_length] for start in starts])
    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    targets_tensor = torch.from_numpy(targets).long().to(device)

    return inputs_tensor, targets_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    should dump all the state from the first three parameters into the file-like object out.
    You can use the state_dict method of both the model and the optimizer to get their relevant states
    and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like object here).
    A typical choice is to have obj be a dictionary,
    but you can use whatever format you want as long as you can load your checkpoint later.
    This function expects the following parameters:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    should load a checkpoint from src (path or filelike object), and then recover the model
    and optimizer states from that checkpoint. Your function should return the iteration number
    that was saved to the checkpoint. You can use torch.load(src) to recover what you saved
    in your save_checkpoint implementation, and the load_state_dict method in both the model
    and optimizers to return them to their previous states.
    This function expects the following parameters:
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    """
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]


if __name__ == "__main__":
    # dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataset = np.arange(10)
    batch_size = 2
    context_length = 3
    device = "cpu"
    res = get_batch(dataset, batch_size, context_length, device)
    print(res)
