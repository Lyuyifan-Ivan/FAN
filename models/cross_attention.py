import math
from tkinter import N
from tkinter.messagebox import NO
from turtle import forward
import torch
import collections
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable


class LayerNorm(nn.Module):
    """
    构建一个LayerNorm Module
    LayerNorm的作用：对x归一化，使x的均值为0，方差为1
    LayerNorm计算公式：x-mean(x)/\sqrt{var(x)+\epsilon} = x-mean(x)/std(x)+\epsilon
    """
    def __init__(self, x_size, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.ones_tensor = nn.Parameter(torch.ones(x_size))
        self.zeros_tensor = nn.Parameter(torch.zeros(x_size))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.ones_tensor * (x - mean) / (std + self.eps) + self.zeros_tensor


def self_attention(query, key, value, dropout=None, mask=None, return_score=False):
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask.cuda()
        score = score.masked_fill(mask==0, -1e9)
    self_atten_softmax = F.softmax(score, dim=-1)
    if dropout is not None:
        self_atten_softmax = dropout(self_atten_softmax)
    if return_score:
        return torch.matmul(self_atten_softmax, value), self_atten_softmax
    return torch.matmul(self_atten_softmax, value)
    #return torch.einsum('ioj,ijk->ijk', [self_atten_softmax, value])


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1, return_score=False) -> None:
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model =d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_softmax = None
        self.return_score = return_score
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        # [batch, seq, d_model] -> [batch, head, seq, d_k]
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask, return_score=self.return_score)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1) -> None:
        """
        :param d_model: FFN第一层输入的维度
        :param d_ff: FNN第二层隐藏层输入的维度
        :param dropout: drop比率
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self, x, res_net=False):
        # x:[batch, seq_len, model_dim]
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        if res_net: # 残差网络
            return self.dropout_2(self.w_2(inter)) + x
        return self.dropout_2(self.w_2(inter))


class SublayerConnecttion(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super(SublayerConnecttion, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))


def clone_module_to_modulelist(module, module_num):
    return nn.ModuleList([deepcopy(module) for _ in range(module_num)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, attn, feed_forward, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnecttion(d_model=d_model, dropout=dropout), 2)
    def forward(self, text, img):
        x = self.sublayer_connection_list[0](text, lambda text:self.attn(text, img, img))
        #x = self.sublayer_connection_list[0](text, lambda text:self.attn(img, text, text))
        return self.sublayer_connection_list[1](x, self.feed_forward)


if __name__ == "__main__":
    bsz = 4
    x = torch.rand(size=(bsz, 80, 768))
    img = torch.rand(size=(bsz, 1, 768))
    attn = MultiHeadAttention(12, 768)
    feed_forward=FeedForward(768,768)
    Cross_atten = EncoderLayer(d_model=768, attn=deepcopy(attn), feed_forward=deepcopy(feed_forward))
    atten_score = Cross_atten(x, img)
    print(atten_score.shape)
