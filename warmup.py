import math

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):

    # 1) linear warmup for warmup_steps iterations
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2) if iteration > max_steps, return min learning rate
    if it > max_steps:
        return min_lr

    # 3) in between, use cosine decay from max_lr down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, "decay_ratio must be between 0 and 1"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff decreases from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)