import torch
import math
import einops.layers.torch as elt

class Attention(torch.nn.Module):
    def __init__(self, embedding_dim = 312,hidden_dim = 256):
        super().__init__()
        self.query_layer = torch.nn.Linear(embedding_dim,hidden_dim)
        self.key_layer = torch.nn.Linear(embedding_dim,hidden_dim)
        self.value_layer = torch.nn.Linear(embedding_dim,hidden_dim)
    
    def forward(self,embedding,mask):
        input_embedding = embedding
        query = self.query_layer(input_embedding)
        key = self.key_layer(input_embedding)
        value = self.value_layer(input_embedding)
        key = elt.Rearrange("b l d -> b d l")(key)#key转置
        attention_prob = torch.matmul(query,key)
        attention_prob += mask * -1e5
        attention_prob = torch.softmax(attention_prob,dim=-1)
        attention_score = torch.matmul(attention_prob,value)
        return (attention_score)
