class GPTConfig:
    def __init__(
        self,
        block_size: int = 1024,  # max sequence length
        vocab_size: int = 50257,  # vocabulary size
        n_layer: int = 12,  # number of layers
        n_head: int = 12,  # number of attention heads
        n_embd: int = 768,  # embedding dimension
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd