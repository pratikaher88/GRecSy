#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys  
sys.path.insert(0, '/Users/pratikaher/DGL/GRecSy/')


# In[22]:


from random import sample
import dgl
import torch
import torch.nn.functional as F
import pandas as pd
from utils import set_random_seed, collate_molgraphs, load_model, collate_movie_graphs, train_test_split_by_time, _split_data
import numpy as np
import torch
import torch.nn as nn
import argparse
from configure import get_exp_configure
import scipy.sparse as sp
from torch.utils.data import DataLoader

from dgllife.utils import EarlyStopping, Meter
from model import compute_loss


# In[6]:


df_rating = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'],  engine='python')
df_user = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/users.dat', sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],  engine='python')
df_movie = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genre'],  engine='python')


# In[67]:


df_temp = df_rating.merge(df_movie, left_on='movie_id', right_on='movie_id', how='left')
df_final = df_temp.merge(df_user, left_on='user_id', right_on='user_id', how='left')


# In[68]:


df_final.head()


# In[94]:


g = dgl.graph((df_final['user_id'].to_numpy(), df_final['movie_id'].to_numpy()))


# In[96]:


isolated_nodes = ((train_graph.in_degrees() == 0) & (train_graph.out_degrees() == 0)).nonzero().squeeze(1)
g = dgl.remove_nodes(train_graph, isolated_nodes)


# In[97]:


g.edata['rating'] = torch.unsqueeze(torch.tensor(df_final['rating'].to_numpy()), dim = 1)
g.ndata['age'] = torch.unsqueeze(torch.tensor(df_user['age'].to_numpy()).float() / 100, dim = 1)


# In[101]:


u, v = g.edges()


# In[102]:


eids = np.arange(train_graph.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)


# In[103]:


train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]


# In[104]:


# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(adj.shape[0], adj.shape[1])
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), train_graph.number_of_edges() // 2)
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]


# In[105]:


train_g = dgl.remove_edges(g, eids[:test_size])


# In[106]:


train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())


# In[112]:


train_pos_g.edata['rating'] = torch.unsqueeze(torch.tensor(df_final['rating'].to_numpy()[:train_size]), dim = 1)


# In[114]:


train_pos_g.ndata['age'] = torch.unsqueeze(torch.tensor(df_user['age'].to_numpy()[:train_size]).float()/100, dim = 1)


# In[107]:


sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
dataloader = dgl.dataloading.EdgeDataLoader(
        train_pos_g, train_pos_g.nodes(), 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=16
        )


# In[116]:


def regress(model, bg, node_subgraph_negative):
    
    bg = bg.to('cpu')
    h = bg.ndata.pop('age')
    e = bg.edata.pop('rating')
    h, e = h.to('cpu', dtype=torch.float), e.to('cpu', dtype=torch.float)

    return model(bg, h, e, node_subgraph_negative)


# In[117]:


from model import MPNNPredictor


# In[122]:


model = MPNNPredictor(node_in_feats=1,edge_in_feats=1,node_out_feats=64,edge_hidden_feats=128,n_tasks=12)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0)

for epoch in range(10):
    
#     with torch.no_grad():
    total_val_loss = 0
    i = 0
    for _, pos_g, neg_g, blocks in dataloader:
        pos_score, neg_score = regress(model, pos_g, neg_g)
        loss = compute_loss(pos_score, neg_score)
#             loss.requires_grad = True
#             print(loss)
        total_val_loss += loss.item()

        if epoch > 0:  # For the epoch 0, no training (just report loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(total_val_loss)
    


# In[ ]:




