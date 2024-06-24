# coding=utf-8
import torch
import math

"""
设定词向量，维度统一为312，词嵌入总数为词库vocab的大小encpder_vocab_size
word_embedding_table = torch.nn.Embedding(num_embeddings=encoder_vocab_size,embedding_dim=312)
encoder_embedding = word_embedding_table(inputs)    传入inputs，就得到了初始的词向量

"""
"""
为了使用输入序列的顺序信息，需要将序列的相对位置和绝对位置注入模型

"""

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model = 312, dropout = 0.05,max_len = 80):
        """
        d_model:编码维度
        dropout:暂退隐藏层的概率
        max_len:语料库中最长单句的长度
        """
        super(PositionalEncoding,self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        parms = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))#优化参数计算
        pe[:,0::2] = torch.sin(position*parms)
        pe[:,1::2] = torch.cos(position*parms)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def foward(self,x):
        x +=self.pe[:,:x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)
