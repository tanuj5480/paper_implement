import torch
import torch.nn as nn
import torch.nn.functional as F


# torch import check
# x = torch.tensor([[1.0,2.0,3.0], [7.0,8.0,1.0]])
# z = F.softmax(x, dim=1)
# print (z)

torch.manual_seed(123)
class CausalSelfAttention(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):

        b, rows, emb_dim = x.shape
        queries = self.W_query(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        attention_scores = (queries @ keys.transpose(1,2)) / (keys.shape[-1] ** 0.5)
        
        # causal mask addition
        seq_len = attention_scores.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # new attention weights after masking
        attentions_weight = F.softmax(masked_attention_scores, dim=1)

        context_vec = attentions_weight @ values

        return context_vec


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

    print ('input vector shape:', dataset.shape)

    attention_layer = CausalSelfAttention(dataset.shape[2], 2)
    output_tensor = attention_layer(dataset)
    print (output_tensor)

    # sample output
    # torch.Size([3, 3, 3])
    # input vector shape: torch.Size([3, 3, 4])
    # tensor([[[-0.0194,  0.0688],
    #      [-0.0947, -0.0051],
    #      [-0.2858, -0.1373]],

    #     [[-0.0179,  0.0657],
    #      [-0.0834, -0.0078],
    #      [-0.2782, -0.1509]],

    #     [[-0.0242,  0.0869],
    #      [-0.0946, -0.0143],
    #      [-0.2743, -0.1590]]], grad_fn=<UnsafeViewBackward0>)