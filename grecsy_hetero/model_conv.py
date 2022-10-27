from typing import Tuple
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import torch

class ConvLayer(nn.Module):

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def __init__(self, in_feats: Tuple[int, int], out_feats: int, dropout: float, aggregator_type: str, norm):
        
        super().__init__()
        self._in_neigh_feats, self._in_self_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.dropout_fn = nn.Dropout(dropout)
        self.norm = norm
        self.fc_self = nn.Linear(self._in_self_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_neigh_feats, out_feats, bias=False)
        self.reset_parameters()

    def forward(self, graph, x):
        
        h_neigh, h_self = x

        h_neigh = self.dropout_fn(h_neigh)
        h_self = self.dropout_fn(h_self)

        graph.srcdata['h'] = h_neigh
        graph.update_all(
            fn.copy_src('h', 'm'),
            fn.mean('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
 
        z = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        z = F.relu(z)

        if self.norm:
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0,
                                torch.tensor(1.).to(z_norm),
                                z_norm)
            z = z / z_norm

        return z



