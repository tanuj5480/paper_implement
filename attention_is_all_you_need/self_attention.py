import torch
import torch.nn as nn
import torch.nn.functional as F


# torch import check
# x = torch.tensor([[1.0,2.0,3.0], [7.0,8.0,1.0]])
# z = F.softmax(x, dim=1)
# print (z)

torch.manual_seed(123)
class SelfAttention(nn.Module):

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

        attention_scores = (queries @ keys.transpose(1,2)) / (emb_dim ** 0.5)
        attentions_weight = F.softmax(attention_scores, dim=1)

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

    attention_layer = SelfAttention(dataset.shape[2], 2)
    output_tensor = attention_layer(dataset)
    print (output_tensor)

    # sample output
    # input vector shape: torch.Size([3, 3, 4])
    # tensor([[[-0.1275, -0.0165],
    #      [-0.1367, -0.0289],
    #      [-0.1357, -0.0281]],

    #     [[-0.1205, -0.0227],
    #      [-0.1299, -0.0355],
    #      [-0.1291, -0.0349]],

    #     [[-0.1236, -0.0162],
    #      [-0.1344, -0.0343],
    #      [-0.1351, -0.0359]]], grad_fn=<UnsafeViewBackward0>)