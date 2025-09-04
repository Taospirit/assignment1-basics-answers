import torch
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


def cross_entropy_impl(logits, labels):
    # Write a function to compute the cross entropy loss,
    # which takes in predicted logits(oi) and targets (xi+1) and computes the cross entropy ℓi = − log softmax(oi)[xi+1]. Your function
    # should handle the following:
    # • Subtract the largest element for numerical stability.
    # • Cancel out log and exp whenever possible.
    # • Handle any additional batch dimensions and return the average across the batch.
    # As with section 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    # log_sum_exp = logits.logsumexp(dim=-1)  # same as above 2 lines
    log_target = logits[torch.arange(logits.shape[0]), labels]
    cross_entropy = -1 * (log_target - log_sum_exp)
    return cross_entropy.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
                return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "m": None,
            "v": None,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.

                t = state.get("t", 0)
                if t == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (v**0.5 + eps)

                # Apply weight decay (decoupled weight decay)
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss


def cosine_learing_rate_scheduler(time_step, max_lr, min_lr, t_warmup, t_iter):
    if time_step < t_warmup:
        return time_step * max_lr / t_warmup
    elif time_step <= t_iter:
        # return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * (time_step - t_warmup) / (t_iter - t_warmup))) / 2
        return min_lr + 0.5 * (
            1 + math.cos(math.pi * (time_step - t_warmup) / (t_iter - t_warmup))
        ) * (max_lr - min_lr)
    else:
        return min_lr


def gradient_clipping(params, max_norm):
    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        return

    total_norm = math.sqrt(sum(p.grad.norm().item() ** 2 for p in params_with_grad))
    if total_norm > max_norm:
        for p in params_with_grad:
            p.grad.data *= max_norm / total_norm


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
