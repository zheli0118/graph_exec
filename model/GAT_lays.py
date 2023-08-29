import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GAT_layer(nn.Module):
    def __init__(self, in_features, out_features, drop, alpha, concat=True):
        super().__init__()
        self.alpha = alpha
        self.in_feature = in_features
        self.out_feature = out_features
        self.drop = drop
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform(self.a.data, gain=1.414)
        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_h, adj):
        h = torch.mm(input_h, self.W)
        N = h.size()[0]
        input_concat = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1). \
            view(N, -1, 2 * self.out_features)
        # input_concat = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_feature)
        e = self.leakyRelu(torch.matmul(input_concat, self.a).squeeze(2))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.drop, training=self.training)
        output_h = torch.matmul(attention, h)
        return output_h

if __name__ == '__main__':
    x=torch.rand(6,10)
    adj=torch.tensor([[random.choice([0,1]) if i!=j else 0 for j in range(6)] for i in range(6)])
    model=GAT_layer(10, 4, 0.2, 0.2)
    print(model(x,adj))