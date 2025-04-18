# alignment-gpt2-pretrain-report

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