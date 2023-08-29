from model.GAT_lays import GAT_layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class MYGAT(nn.Module):
    def __init__(self, input_feature_size, output_size, nclass, dropout, alpha, nheads):
        super().__init__()
        self.nheads = nheads
        self.alpha = alpha
        self.dropout = dropout
        self.nclass = nclass
        self.output_size = output_size
        self.input_feature_size = input_feature_size
        self.attentions = [GAT_layer(input_feature_size, output_size, dropout, alpha) for _ in range(nheads)]
        for i, attention in  enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_attention = GAT_layer(in_features=input_feature_size*nheads,
                                       out_features=output_size,
                                       drop=dropout, alpha=alpha)
    def forward(self,x,adj):
        x = F.dropout(x,self.dropout,training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions],dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attention(x, adj))
        return F.log_softmax(x, dim=1)