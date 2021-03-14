import torch
import math, copy
import torch.nn.functional as F
import torch.nn as nn

from utils import clones

def attention(query, key, value, mask=None, dropout=None):   #query:[batch_size, ]
    d_k = query.size(-1)
    #假设d_k和d_v相等

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, head_nums, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # 保证d_model即维数可以被头的数量head_nums整除
        assert d_model % head_nums == 0


        self.d_k = int(d_model / head_nums)
        self.head_nums = head_nums
        self.linears = clones(nn.Linear(d_model, d_model), 4)  #初始化了4个，前三个用于Q,K,V向量化，最后一个用于MultiHead-Attention最后的部分
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        ## 1) Do all the linear projections in batch from    d_model => [h, d_k]

        query, key, value = [layer(data).view(batch_size, -1, self.head_nums, self.d_k) for layer, data in zip(self.linears, (query, key, value))]
        #query, key, value ==> [batch_size, -1, head_nums, d_k]  #分成了多个头

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_nums * self.d_k)
        #调用view之前最好先contiguous
        return self.linears[-1](x)