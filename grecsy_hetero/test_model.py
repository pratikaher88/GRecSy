import sys  
sys.path.insert(0, '/Users/pratikaher/DGL/GRecSy/')

from random import sample
import dgl
import torch
import pandas as pd
import numpy as np
import torch
from grecsy_hetero.model import compute_loss

from dgl import AddReverse
transform = AddReverse()

df_rating = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'],  engine='python')
df_user = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/users.dat', sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],  engine='python')
df_movie = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genre'],  engine='python')

# print(df_rating.head())
# print(df_movie.head())
# print(df_user.head())

df_temp = df_rating.merge(df_movie, left_on='movie_id', right_on='movie_id', how='left')
df_final = df_temp.merge(df_user, left_on='user_id', right_on='user_id', how='left')

print(df_user.shape)

# df_final = df_final[["user_id","movie_id","rating","age"]]

df_movie['release_year'] = df_movie.title.str.extract("\((\d{4})\)", expand=True).astype(str)
df_movie['release_year'] = pd.to_datetime(df_movie.release_year, format='%Y')
df_movie['release_year'] = df_movie.release_year.dt.year
df_movie['title'] = df_movie.title.str[:-7]

hetero_graph = dgl.heterograph({('user', 'rates', 'movie'): (df_final['user_id'].to_numpy(), df_final['movie_id'].to_numpy())})

hetero_graph = transform(hetero_graph)

print(hetero_graph)

# print(len(list(set(df_final['movie_id'].to_numpy()))))

isolated_nodes = ((hetero_graph.in_degrees(etype='rates') == 0) & (hetero_graph.out_degrees(etype='rev_rates') == 0)).nonzero().squeeze(1)
hetero_graph = dgl.remove_nodes(hetero_graph, isolated_nodes, ntype='movie')

isolated_nodes = ((hetero_graph.in_degrees(etype='rev_rates') == 0) & (hetero_graph.out_degrees(etype='rates') == 0)).nonzero().squeeze(1)
hetero_graph = dgl.remove_nodes(hetero_graph, isolated_nodes, ntype='user')


# hetero_graph.remove_nodes(torch.tensor([0]), ntype='user')

# hetero_graph.nodes['user'].data['features'] = torch.unsqueeze(torch.tensor(df_user['age'].to_numpy()), dim = 1).float()

# # Fix this to include actual release years
# hetero_graph.nodes['movie'].data['release_year'] = torch.tensor(df_movie['release_year'].values).float()
# hetero_graph.nodes['movie'].data['features'] = torch.randn(3953, 1).float()
# hetero_graph.edges['rates'].data['rating'] =  torch.unsqueeze(torch.tensor(df_final['rating'].to_numpy()), dim = 1)


df_movie_modified = df_movie[df_movie['movie_id'].isin(list(set(df_final['movie_id'].to_numpy())))]
df_user_modified = df_user[df_user['user_id'].isin(list(set(df_final['user_id'].to_numpy())))]

# print(df_final.head())
# print(df_movie_modified.shape, df_user_modified.shape)

hetero_graph.nodes['user'].data['features'] = torch.unsqueeze(torch.tensor(df_user_modified['age'].to_numpy()), dim = 1).float()
hetero_graph.nodes['movie'].data['features'] = torch.unsqueeze(torch.tensor(df_movie_modified['release_year'].to_numpy()), dim = 1).float()
hetero_graph.edges['rates'].data['rating'] =  torch.unsqueeze(torch.tensor(df_final['rating'].to_numpy()), dim = 1)


# print(sorted(hetero_graph.successors( 1, etype='rates').numpy()))
# print(sorted(list(df_final.loc[df_final['user_id'] == 1]['movie_id'])))

# import sys
# sys.exit()

from grecsy_hetero.model import MPNNConvModel

params = {'hidden_dim' : 128, 'out_dim' : 64 }

dim_dict = {'user': hetero_graph.nodes['user'].data['features'].shape[1],
            'movie': hetero_graph.nodes['movie'].data['features'].shape[1],
            'out': params['out_dim'],
            'hidden': params['hidden_dim']}


## train test split

# solve issue with reverse edges

# eids = np.arange(hetero_graph.number_of_edges())
# eids = np.random.permutation(eids)
# test_size = int(len(eids) * 0.2)
# validation_size = int(len(eids) * 0.1)
# train_size = hetero_graph.number_of_edges() - test_size - validation_size

# hetero_graph = dgl.remove_edges(hetero_graph, eids[:test_size+validation_size], etype='rates')
# hetero_graph = dgl.remove_edges(hetero_graph, eids[:test_size+validation_size], etype='rev-rates')

## creating dataloaders

train_eids_dict = {}
valid_eids_dict = {}

eids = np.arange(hetero_graph.number_of_edges(etype='rates'))
eids = np.random.permutation(eids)

test_size = int(len(eids) * 0.1)
valid_size = int(len(eids) * 0.1)
train_size = len(eids) - test_size - valid_size

print(train_size)

for e in hetero_graph.etypes:
    train_eids_dict[e] = eids[:train_size]
    valid_eids_dict[e] = eids[train_size:train_size+valid_size]

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
dataloader = dgl.dataloading.EdgeDataLoader(
        hetero_graph, train_eids_dict, 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=16
        )

valid_dataloader = dgl.dataloading.EdgeDataLoader(
        hetero_graph, valid_eids_dict, 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=16
        )

model = MPNNConvModel(hetero_graph, dim_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0)

total_loss = 0

for _, pos_g, neg_g, blocks in dataloader:
    
    optimizer.zero_grad()

    input_features = blocks[0].srcdata['features']

    # print("INPUT",input_features['movie'].shape)
    
    # print("input features", blocks[0].srcdata['release_year']['movie'].shape )

    # input_features = {'user': blocks[0].srcdata['age']['user'].float(),
    #                 # 'movie' : blocks[0].srcdata['release_year']['movie'].float()
    #                 }

    _, pos_score, neg_score = model(blocks,
                                    input_features,
                                    pos_g,
                                    neg_g)

    print(pos_score, neg_score)

    loss = compute_loss(pos_score, neg_score)
    # print("Loss output", loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()



print(total_loss)

# validation score

model.eval()
with torch.no_grad():
    total_val_loss = 0

    for _, pos_g, neg_g, blocks in valid_dataloader:

        optimizer.zero_grad()

        input_features = blocks[0].srcdata['features']

        _, pos_score, neg_score = model(blocks,
                                input_features,
                                pos_g,
                                neg_g)

        # print(pos_score, neg_score)
        loss = compute_loss(pos_score, neg_score)
        total_val_loss += loss.item()


print(total_val_loss)
    

