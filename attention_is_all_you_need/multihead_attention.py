from causal_self_attention import CausalSelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


if __name__ == "__main__":
    
    base_matrix = torch.tensor(
        [[0.1, 0.2, 0.1, 0.5],   # Token 0 ("the")
        [0.8, 0.9, 0.1, 0.0],   # Token 1 ("dog")
        [0.9, 0.8, 0.2, 0.1]]   # Token 2 ("barked")
    )    

    batch_1 = base_matrix.clone()
    batch_2 = base_matrix + torch.randn_like(base_matrix) * 0.05 
    batch_3 = base_matrix + torch.randn_like(base_matrix) * 0.1

    dataset = torch.stack((batch_1,batch_2,batch_3), dim=0)
    context_length = dataset.shape[1] # This is the number of tokens
    print ('input vector shape:', dataset.shape)

    d_in, d_out = dataset.shape[2], 2
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
    )

    context_vecs = mha(dataset)    
    print (context_vecs)

    # sample output
    # input vector shape: torch.Size([3, 3, 4])
    # tensor([[[-0.0194,  0.0688,  0.0738,  0.0402],
    #      [-0.0947, -0.0051,  0.2431,  0.2501],
    #      [-0.2858, -0.1373,  0.5887,  0.6340]],

    #     [[-0.0179,  0.0657,  0.0745,  0.0414],
    #      [-0.0834, -0.0078,  0.2390,  0.2520],
    #      [-0.2782, -0.1509,  0.5754,  0.6308]],

    #     [[-0.0242,  0.0869,  0.0886,  0.0507],
    #      [-0.0946, -0.0143,  0.2476,  0.2443],
    #      [-0.2743, -0.1590,  0.6011,  0.6798]]], grad_fn=<CatBackward0>)