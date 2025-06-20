import tiktoken
import torch
import os
import numpy as np
 
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
 
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, master_process, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
 
        # at init load tokens from disk and store them in memory
        assert split in {"train", "val"}, f"Split must be either 'train' or 'val', got {split}"

        # Get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert shards, f"No shards found for split {split}"

        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
 
        # state
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
 
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y