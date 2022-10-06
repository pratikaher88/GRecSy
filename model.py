# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPNN
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
import dgl.function as fn

from dgl.nn.pytorch import Set2Set

from dgllife.model.gnn import MPNNGNN

__all__ = ['MPNNPredictor']

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

# pylint: disable=W0221
class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )
        self.predictor = DotProductPredictor()

    def forward(self, g, node_feats, edge_feats, node_subgraph_negative):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        # graph_feats = self.readout(g, node_feats)
        return self.predictor(g, node_feats), self.predictor(node_subgraph_negative, node_feats)
        # return node_feats
        
        # graph_feats = self.readout(g, node_feats)
        # return self.predict(graph_feats)



def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# def max_margin_loss(pos_score,
#                     neg_score,
#                     delta: float,
#                     neg_sample_size: int,
#                     use_recency: bool = False,
#                     recency_scores=None,
#                     remove_false_negative: bool = False,
#                     negative_mask=None,
#                     cuda=False,
#                     device=None
#                     ):
#     """
#     Simple max margin loss.

#     Parameters
#     ----------
#     pos_score:
#         All similarity scores for positive examples.
#     neg_score:
#         All similarity scores for negative examples.
#     delta:
#         Delta from which the pos_score should be higher than all its corresponding neg_score.
#     neg_sample_size:
#         Number of negative examples to generate for each positive example.
#         See main.SearchableHyperparameters for more details.
#     use_recency:
#         If true, loss will be divided by the recency, i.e. more recent positive examples will be given a
#         greater weight in the total loss.
#         See main.SearchableHyperparameters for more details.
#     recency_score:
#         Loss will be divided by the recency if use_recency == True. Those are the recency, for all training edges.
#     remove_false_negative:
#         When generating negative examples, it is possible that a random negative example is actually in the graph,
#         i.e. it should not be a negative example. If true, those will be removed.
#     negative_mask:
#         For each negative example, indicator if it is a false negative or not.
#     """
#     all_scores = torch.empty(0)
#     if cuda:
#         all_scores = all_scores.to(device)
#     for etype in pos_score.keys():
#         neg_score_tensor = neg_score[etype]
#         pos_score_tensor = pos_score[etype]
#         neg_score_tensor = neg_score_tensor.reshape(-1, neg_sample_size)
#         if remove_false_negative:
#             negative_mask_tensor = negative_mask[etype].reshape(-1, neg_sample_size)
#         else:
#             negative_mask_tensor = torch.zeros(size=neg_score_tensor.shape)
#         if cuda:
#             negative_mask_tensor = negative_mask_tensor.to(device)
#         scores = neg_score_tensor + delta - pos_score_tensor - negative_mask_tensor
#         relu = nn.ReLU()
#         scores = relu(scores)
#         if use_recency:
#             try:
#                 recency_scores_tensor = recency_scores[etype]
#                 scores = scores / torch.unsqueeze(recency_scores_tensor, 1)
#             except KeyError:  # Not all edge types have recency. Only training edges have recency (i.e. clicks & buys)
#                 pass
#         all_scores = torch.cat((all_scores, scores), 0)
#     return torch.mean(all_scores)