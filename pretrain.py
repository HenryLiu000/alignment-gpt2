from config import GPTConfig
from model import GPT2
from warmup import get_lr
from dataloader import DataLoaderLite
import torch
import time
import math



total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 512 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
 
dataloader = DataLoaderLite(B=B, T=T)
# # get a small sample of text to train on
# with open('input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]  # Truncate to 1000 characters
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode(text)

# B, T = 4, 32  # Batch size and sequence length
# buf = torch.tensor(tokens[:B * T + 1])  # Buffer for the batch

# x = buf[:-1].view(B, T)  # Input tensor of shape (B, T)
# y = buf[1:].view(B, T)  # Target tensor of shape (B, T)


# initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(GPTConfig())
model.to(device)
# model = torch.compile(model)

# optimize the model

model.train()



max_lr = 6e-4 
min_lr = max_lr * 0.1 
warmup_steps = 10 
max_steps = 50 

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # fused AdamW optimizer

# torch.set_float32_matmul_precision('high') # TF32 matmul precision

for step in range(max_steps):  # Number of epochs
    t0 = time.time()
    optimizer.zero_grad()


    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping

    lr = get_lr(step, 
                max_lr=max_lr, 
                min_lr=min_lr, 
                warmup_steps=warmup_steps, 
                max_steps=max_steps)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
    t1 = time.time()
    dt = (t1 - t0)  # seconds per step
    tokens_processed = dataloader.B * dataloader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    # dt ms
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
