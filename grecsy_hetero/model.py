from statistics import mean
from unittest import result
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from model_conv import ConvLayer
import torch.nn.functional as F
import torch

def max_margin_loss(pos_score, neg_score, neg_sample_size=45):

    all_scores = torch.empty(0)

    for etype in pos_score.keys():
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]
        # print(pos_score_tensor.shape, neg_score_tensor.shape)
        neg_score_tensor = neg_score_tensor.reshape(pos_score_tensor.shape[0], -1)
        # print(neg_score_tensor.shape)
        scores = neg_score_tensor - pos_score_tensor
        relu = nn.ReLU()
        scores = relu(scores)
        all_scores = torch.cat((all_scores, scores), 0)

    return torch.mean(all_scores)


def compute_loss(pos_score, neg_score):
    # understand tensor shapes
    # understand cross entropy

    all_scores = 0

    for etype in pos_score:
        
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]

        pos_score_tensor, neg_score_tensor = pos_score_tensor.squeeze(), neg_score_tensor.squeeze()

        # print("INPUT TENSORS",pos_score_tensor, neg_score_tensor)

        # if pos_score_tensor.get_shape[1] > 0 and neg_score_tensor.get_shape[1] > 0:
        scores = torch.cat([pos_score_tensor, neg_score_tensor])
        labels = torch.cat([torch.ones(pos_score_tensor.shape[0]), torch.zeros(neg_score_tensor.shape[0])])

        # print(F.binary_cross_entropy_with_logits(scores, labels))

        all_scores += F.binary_cross_entropy_with_logits(scores, labels)

        # all_scores = torch.cat((all_scores, result), 0)

    return all_scores
    # return torch.mean(all_scores)

        # print(BCEL)

        # print(all_scores)

        # all_scores = torch.cat((all_scores, BCEL), dim = 1)
        # print(all_scores)

    # return torch.mean(all_scores)

class CosinePrediction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, graph, h):
        with graph.local_scope():
            for etype in graph.canonical_etypes:
                # try:
                graph.nodes[etype[0]].data['norm_h'] = F.normalize(h[etype[0]], p=2, dim=-1)
                graph.nodes[etype[2]].data['norm_h'] = F.normalize(h[etype[2]], p=2, dim=-1)
                graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=etype)
                # except KeyError:
                #     pass  # For etypes that are not in training eids, thus have no 'h'
            ratings = graph.edata['cos']

        return ratings

class MPNNConvModel(nn.Module):

    def __init__(self, input_graph, dim_dict, n_layers = 3, dropout = 0, aggregator_hetero = 'sum', norm = True):

        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype[1]: ConvLayer((dim_dict[etype[0]], dim_dict[etype[2]]), dim_dict['hidden'], dropout,
                                        aggregator_hetero, norm)
                    for etype in input_graph.canonical_etypes},
                aggregate=aggregator_hetero)
        )

        for i in range(n_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype[1]: ConvLayer((dim_dict['hidden'], dim_dict['hidden']), dim_dict['hidden'], dropout,
                                         aggregator_hetero, norm)
                     for etype in input_graph.canonical_etypes},
                    aggregate=aggregator_hetero))

        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype[1]: ConvLayer((dim_dict['hidden'], dim_dict['hidden']), dim_dict['out'], dropout,
                                     aggregator_hetero, norm)
                 for etype in input_graph.canonical_etypes},
                aggregate=aggregator_hetero))

        self.pred_fn = CosinePrediction()
    
    def get_repr(self, blocks, h):
        for i in range(len(blocks)):
            layer = self.layers[i]
            # print("H before", h)
            h = layer(blocks[i], h)
            # print("H after", h)
        return h

    def forward(self, blocks, h, pos_g, neg_g, embedding_layer = True):
        
        h = self.get_repr(blocks, h)

        pos_score = self.pred_fn(pos_g, h)
        neg_score = self.pred_fn(neg_g, h)
        return h, pos_score, neg_score

