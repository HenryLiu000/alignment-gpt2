# alignment-gpt2-pretrain-report

The report is to introduce the pretraining of gpt2 model. I will give the detailed explanation of every git commit in the report and I will also illstrate all the important steps of pretraining gpt2 model.

## 1. create readme: 
This is the first commit, which is to create the readme file.

## 2. config: 
This commit is to create a config class for the gpt2 model, which contains the hyperparameters of the gpt2 structure. Having a config class makes it convenient to change the hyperparameters of the model. The config class is in config.py file.

## 3. model structure: 
This commit is to create the gpt2 model structure. The 124M GPT2 consists of a word token embedding layer(wte), a word position embedding layer(wpe), 12 transformer blocks, a layer normalization layer and a linear layer. Following the  tutorial from Karparthy, I implemented the gpt2 model structure in the following steps:

1. **The overall structure of the gpt2 model**: This is implemented in class GPT2, in the init function, I use nn.Embedding to create wte and wpe, nn.LayerNorm to create ln_f and a nn.Linear layer for the output layer. The only complicated part is the transformer block. I use the nn.ModuleList to create the 12 blocks and leave the block class to be defined in the next step. For the forward function of GPT2 class, I follow Karparthy's tutorial and make the input as a (B, T) tensor. For each row, the T numbers are the index of the tokens in the vocabulary. The forward function converts the index to the corresponding embeddings and processes them through the gpt2 model. The output is then passed through the linear layer to produce the final logits and the logits are fed into the cross entropy loss function if training.

2. **The transformer block**: The transformer block is implemented in the class Block. The Block class contains the following components:
   - **ln_1**: This is a layer normalization layer implemented using nn.LayerNorm.
   - **attn**: This is the attention layer, which is implemented in the next step.
   - **ln_2**: This is another layer normalization layer implemented using nn.LayerNorm.
   - **mlp**: This is a multi-layer perceptron, which is implemented using nn.Sequential containing two linear layers and a GELU activation function.

For the forward function, it is trivial, if the data is represented as x, then we just need to feed x to the attention layer and the mlp, with layer normalization and residual connections applied to both the outputs.

3. **The attention layer**: The attention layer is implemented in the class CausalAttention. The attention layer is fed with the input x and needs to compute the attention scores and the attention output. The most important thing is how to create the right attention mask. The attention mask is a lower triangular matrix, which won't participate the gradient flow during backpropagation. Because of this, I use the torch.tril function to create the attention mask and put the attention mask into the register_buffer, which is a buffer that won't participate in the gradient flow. The implementation of the forward function is as follows:
    - First, I compute the attention scores for multiple heads using the query, key and value matrices. The attention scores are computed using the formula: Q * K^T / sqrt(d_k), where Q is the query matrix, K is the key matrix and d_k is the dimension of the key matrix.
    - Then, I apply the attention mask to the attention scores. Since we have already created a lower triangular matrix, we can simply extract the first T rows and T columns (T is the sequence length) of the attention mask, which is a lower triangular matrix, and then apply the attention mask to the attention scores. It means that the attention scores for the future tokens are set to -inf and it makes sure that the attention scores for the future tokens won't participate in the gradient flow.
    - Finally, I compute the attention output using the softmax function and the value matrix for every head. The attention output is computed using the formula: softmax(Q * K^T / sqrt(d_k)) * V, where V is the value matrix. The attention output is then concatenated and projected to the original dimension using the linear layer. The output is then returned as the final result of the forward function.

