import torch
import torch.nn.functional as F


# torch import check
# x = torch.tensor([[1.0,2.0,3.0], [7.0,8.0,1.0]])
# z = F.softmax(x, dim=1)
# print (z)


class SelfAttention(nn.module):

    def __init__(self, d_in, d_out, qkv_bias, emb_dim):
        self.d_in = d_in
        self.d_out = d_out
        self.emb_dim = emb_dim

        self.W_query = nn.Linear(d_in, d_out, qkv_bias = False)
        self.W_keys = nn.Linear(d_in, d_out, qkv_bias = False)
        self.W_values = nn.Linear(d_in, d_out, qkv_bias = False)

    def forward(self, x):

        b, rows, emb_dim = x.shape
        queries = self.W_query(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        attention_scores = queries @ keys.T
        attentions_weight = F.softmax(attention_scores, dim=1)

        context_vec = attentions_weight @ values

        return context_vec



