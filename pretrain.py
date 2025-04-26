from config import GPTConfig
from model import GPT2
from warmup import get_lr
from dataloader import DataLoaderLite
import torch
import time
import math



dataloader = DataLoaderLite(8, 512)
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

for epoch in range(max_steps):  # Number of epochs
    t0 = time.time()
    optimizer.zero_grad()
    x, y = dataloader.next_batch()
    x = x.to(device)
    y = y.to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        #import code; code.interact(local=locals())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping

    lr = get_lr(epoch, 
                max_lr=max_lr, 
                min_lr=min_lr, 
                warmup_steps=warmup_steps, 
                max_steps=max_steps)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000  # Convert to milliseconds
    print(f"Epoch {epoch}: loss = {loss.item()}", f"dt = {dt:.2f} ms")
