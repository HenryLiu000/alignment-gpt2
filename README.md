*# alignment-gpt2-pretrain-report



The report is to introduce the pretraining of gpt2 model. I will give the detailed explanation of every git commit in the report and I will also illstrate all the important steps of pretraining gpt2 model.

## Section 1
### 1. create readme: 
 This is the first commit, which is to create the readme file.

### 2. config: 
 This commit is to create a config class for the gpt2 model, which contains the hyperparameters of the gpt2 structure. Having a config class makes it convenient to change the hyperparameters of the model. The config class is in config.py file.

### 3. model structure: 
 This commit is to create the gpt2 model structure. The 124M GPT2 consists of a word token embedding layer(wte), a word position embedding layer(wpe), 12 transformer blocks, a layer normalization layer and a linear layer. Following the  tutorial from Karparthy, I implemented the gpt2 model structure in the following steps:

1. **The overall structure of the gpt2 model**: 
   This is implemented in class GPT2, in the init function, I use nn.Embedding to create wte and wpe, nn.LayerNorm to create ln_f and a nn.Linear layer for the output layer. The only complicated part is the transformer block. I use the nn.ModuleList to create the 12 blocks and leave the block class to be defined in the next step. For the forward function of GPT2 class, I follow Karparthy's tutorial and make the input as a (B, T) tensor. For each row, the T numbers are the index of the tokens in the vocabulary. The forward function converts the index to the corresponding embeddings and processes them through the gpt2 model. The output is then passed through the linear layer to produce the final logits and the logits are fed into the cross entropy loss function if training.

2. **The transformer block**: 
    The transformer block is implemented in the class Block. The Block class contains the following components:
      - **ln_1**: This is a layer normalization layer implemented using nn.LayerNorm.
      - **attn**: This is the attention layer, which is implemented in the next step.
      - **ln_2**: This is another layer normalization layer implemented using nn.LayerNorm.
      - **mlp**: This is a multi-layer perceptron, which is implemented using nn.Sequential containing two linear layers and a GELU activation function.

   For the forward function, it is trivial, if the data is represented as x, then we just need to feed x to the attention layer and the mlp, with layer normalization and residual connections applied to both the outputs.

3. **The attention layer**: 
   This is implemented in the class CausalAttention. The attention layer is fed with the input x and needs to compute the attention scores and the attention output. The most important thing is how to create the right attention mask. The attention mask is a lower triangular matrix, which won't participate in the gradient flow during backpropagation. Because of this, I use the torch.tril function to create the attention mask and put the attention mask into the register_buffer, which is a buffer that won't participate in the gradient flow. The implementation of the forward function is as follows:
    - First, I compute the attention scores for multiple heads using the query, key and value matrices. The attention scores are computed using the formula: $\frac{QK^T}{\sqrt{d_k}}$, where $Q$ is the query matrix, $K$ is the key matrix and $d_k$ is the dimension of the key matrix.
    - Then, I apply the attention mask to the attention scores. Since we have already created a lower triangular matrix as the attention mask, we can simply extract the first T rows and T columns (T is the sequence length) of the attention mask and then apply it to the attention scores. It means that the attention scores for the future tokens are set to -inf and it makes sure that the attention scores for the future tokens won't participate in the gradient flow.
    - Finally, I compute the attention output using the softmax function and the value matrix for every head. The attention output is computed using the formula: $softmax(\frac{QK^T}{\sqrt{d_k}}) * V$, where $V$ is the value matrix. The attention output is then concatenated and projected to the original dimension using the linear layer. The output is then returned as the final result of the forward function.

### 4. weight sharing scheme:
 This commit is to implement the weight sharing scheme. The weight of wte of the transformer should actually be similar to the weight of the output layer. So I just let these two weight matrices to be the same.

### 5. initializing weight for a new model:

 This commit is to implement the weight initialization for a new model. Since the task is to pretrain a new model and I have no GPU in my local computer so I will not try load state dict from huggingface to test the model's performance. To initialize the weight of the model, I follow Karparthy's tutorial, which is also the weight initialization method used in the huggingface library. The weight initialization method is as follows:
1. **The embedding layer**: 
   The weight of the embedding layer is initialized using the normal distribution with mean 0 and standard deviation 0.02. The weight of the embedding layer is a matrix of size (vocab_size, d_model), where vocab_size is the size of the vocabulary and d_model is the dimension of the model.
2. **The linear layer**: 
   The weight of the linear layer is initialized using the normal distribution with mean 0 and standard deviation 0.02. If the layer has bias, the bias is initialized to zero. And if the layer has the additional feature named `NANOGPT_SCALE_INIT`, the standard deviation is scaled.

### 6. initialize model and load a small batch of data:

 This commit is just to initialize a model and load a small batch of data from input.txt

### 7. training loop:

 Within a small batch of data, I can write a training loop to train the model. Simply feed the small batch of data to the model and do backpropagation.

### 8. dataloader:

 In order to feed the data to the model in an efficient way, I need to implement a dataloader. The dataloader is implemented in the class DataloaderLite in dataloader.py.

## Section 2

### 9. Test time

 This commit is to implement the code for testing the training time of the model. I cannot test the training time though. However, I will outline the steps to measure the training time effectively.

### 10. TF32

 This commit is to use TF32 as the default floating point type for the training. TF32 is a new floating point type introduced by NVIDIA in the Ampere architecture. It is a 19-bit floating point type with 8 bits for the exponent and 10 bits for the mantissa. When training the model, the GPU will keep an accumulator with FP32 precision. The gradients will be computed in TF32 precision and then accumulated in FP32 precision. This is just done by setting the `torch.set_float32_matmul_precision('high')`. This allows for faster training while maintaining model accuracy.

### 11. BF16
 This commit is to use BF16 as the default floating point type for the training. I just use the following code to set the default floating point type to BF16:
 ```python
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        # training logic here
        pass
 ```

 I test the time on a rented Geforce RTX 3050 Ti GPU. Since the Geforce RTX 3050 Ti is also Ampere architecture, the TF32 and BF16 are supported. I set the batch size to 8 and the sequence length to 512. If I use the FP32 precision for training, the training time for a batch is about 9s. If I use the TF32 precision for training, the training time for a batch is about 6s. If I use the BF16 precision for training, the training time for a batch is about 4s. So mixed precision training is really helpful for speeding up the training process.

### 12. torch.compile
 This commit introduces torch.compile, a feature in PyTorch 2.0 and later designed to further accelerate model training and inference. torch.compile works by converting PyTorch code into optimized low-level kernels using techniques like graph capture and kernel fusion. This process reduces Python overhead and minimizes GPU memory access, often leading to significant speedups beyond what TF32 or BF16 alone provide. However, the GPU does not have many SMs, so I cannot run torch.compile on the GPU.

### 13. Flash Attention

 This commit focuses on optimizing the core attention mechanism itself using Flash Attention. Standard self-attention (as implemented in commit #3) has quadratic time and memory complexity with respect to sequence length ($O(N^2)$). This is because it needs to compute and store a large $N \times N$ attention score matrix, leading to significant memory usage and slow reads/writes between the GPU's fast on-chip SRAM and slower High Bandwidth Memory (HBM).

 Flash Attention is an optimized implementation that avoids explicitly forming the large attention matrix in HBM. It uses techniques like tiling (processing $Q, K, V$ in blocks), kernel fusion, and careful management of reads/writes between SRAM and HBM to compute the exact same attention output much faster and with linear memory complexity ($O(N)$).

### 14. nice numbers

 In consideration of the structure of GPUs, I need to set the hyperparameters of the model to be nice numbers. For example, the token embedding numbers should be a multiple of 8, so I change it to 50304.

## Section 3

### 15. AdamW parameter and gradient clipping

 In order to train the model with stability, I need to follow the report of GPT3 to use their AdamW parameters and also use gradient clipping.

### 16. warmup learning rate and cosine decay

 The training of the model uses the warmup learning rate and cosine decay. The warmup learning rate is a learning rate that starts from 0 and increases to the maximum learning rate in the first few steps. Then the learning rate decays to 0 using the cosine decay, that is, there is a cosine coefficient that is declining over the training steps from 1 to 0. The cosine coefficient is then used to control the learning rate from the maximum learning rate to the minimum learning rate. The code is available in the file `warmup.py`. 

### 17. Weight Decay and Fused AdamW

 In this section, two key optimizations are introduced: selective weight decay and the use of a fused AdamW optimizer. Weight decay serves as a regularization strategy to mitigate overfitting by penalizing large weights. However, parameters such as the scale factors in layer normalization and the biases in linear layers, which typically have only a single value, are excluded from weight decay. By classifying parameters into those that require weight decay and those that do not, more effective regularization is achieved.

Furthermore, the fused AdamW optimizer is employed to improve efficiency. When the "fused" parameter is set to True, the optimizer leverages fused kernels to compute gradients, update weights, and perform additional operations. This approach reduces both memory accesses and the number of kernel launches, ultimately decreasing overhead and accelerating training. These enhancements are implemented in the `configure_optimizers()` function within the `GPT2` class in `model.py`.

### 18. Gradient Accumulation

 Gradient accumulation is a technique used to effectively increase the batch size without requiring additional GPU memory. This is particularly useful when training large models or when GPU memory is limited. The idea is to perform multiple forward and backward passes with smaller batches, accumulating the gradients over these passes, and then update the model weights only once after a specified number of iterations. This allows for simulating a larger batch size while keeping the memory footprint manageable. The implementation of gradient accumulation is straightforward: during each iteration, gradients are computed and accumulated, but the model weights are only updated after a specified number of iterations, controlled by the `grad_accum_steps` parameter. This is done in the training loop, where the optimizer's step function is called only after the specified number of iterations.

### 19. DDP (Distributed Data Parallel)

 This section introduces Distributed Data Parallel (DDP), a technique designed to significantly accelerate model training by leveraging multiple GPUs. Its core principle is **data parallelism** coupled with **gradient synchronization**. DDP operates by first replicating the **entire model** onto each participating GPU. Subsequently, a large data batch is divided, and **each GPU processes a distinct data shard** independently, performing forward and backward passes to compute local gradients. Critically, **before the optimizer updates the model weights**, DDP automatically **synchronizes and aggregates** (typically by averaging) these locally computed gradients across all GPUs involved. Finally, **every model replica is updated using the exact same aggregated gradient**, ensuring parameter consistency across all devices while parallelizing the workload, which drastically reduces overall training time.

### 20. FineWeb Dataset
 To implement the final training, we need to switch to a larger dataset. FineWeb is a very large dataset (around 15 trillion tokens) created by Hugging Face. It's derived from the Common Crawl web scrapes (snapshots of the public internet). Unlike raw web data, FineWeb has undergone extensive filtering and deduplication processes. The goal was to create an open dataset comparable in quality to the large, proprietary datasets used to train state-of-the-art closed models (like earlier versions of GPT). Unfortunately, I could download the dataset due to the capacity limit of my disk, but I finish all the code to use this dataset to pretrain GPT2, the downloading code is in `fineweb.py` and the dataset used for training has been switched to this dataset in `dataloader.py` and `pretrain.py`.

### 21. validation split
 The validation split, a crucial held-out subset of the total dataset not used for direct model training, plays a vital monitoring role in developing models like GPT-2. Its primary purpose is to offer an unbiased estimate of the model's performance on unseen data, which is essential for detecting overfitting by observing if validation loss increases while training loss decreases. Furthermore, validation metrics guide hyperparameter tuning by indicating which settings yield the best generalization, inform early stopping strategies to prevent excessive training and save resources, and facilitate model checkpointing by saving the model version that performs best on this unseen data. In practice, this involves periodically evaluating the model (e.g., using `model.eval()` in PyTorch) on the validation data loader—created by splitting the main dataset like FineWeb—to calculate metrics such as loss and perplexity without updating model weights, ultimately helping to train a more robust and better-generalizing model rather than one that simply memorizes training examples.

### 22. Hellaswag

 Hellaswag (often stylized HellaSwag) is a popular and challenging benchmark dataset specifically designed to evaluate the **commonsense reasoning** capabilities of artificial intelligence models, particularly large language models (LLMs). It falls under the category of **Commonsense Natural Language Inference (NLI)**. I use this dataset to evaluate the capabilities of the pretrained model. Similar to the situation of the FineWeb dataset, I cannot download due to the limit of my disk, but I have prepared the code to download and use this dataset.