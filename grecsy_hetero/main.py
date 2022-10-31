import pickle
import create_graph
import numpy as np
import dgl
import torch
from model import compute_loss, max_margin_loss

base_path = '/Users/pratikaher/DGL/GRecSy/saved_files/'
epochs = 10

# hetero_graph = create_graph.create_graph()

# pickle down the graph for later use
# pickle.dump(hetero_graph, open(base_path+'pickled_graph.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

hetero_graph = pickle.load(open(base_path+'pickled_graph.pickle', 'rb'))

print(hetero_graph)

from model import MPNNConvModel

params = {'hidden_dim' : 128, 'out_dim' : 64 }

dim_dict = {'user': hetero_graph.nodes['user'].data['features'].shape[1],
            'movie': hetero_graph.nodes['movie'].data['features'].shape[1],
            'out': params['out_dim'],
            'hidden': params['hidden_dim']}

train_eids_dict = {}
valid_eids_dict = {}

eids = np.arange(hetero_graph.number_of_edges(etype='rates'))
eids = np.random.permutation(eids)

test_size = int(len(eids) * 0.1)
valid_size = int(len(eids) * 0.1)
train_size = len(eids) - test_size - valid_size

for e in hetero_graph.etypes:
    train_eids_dict[e] = eids[:train_size]
    valid_eids_dict[e] = eids[train_size:train_size+valid_size]

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])

train_dataloader = dgl.dataloading.EdgeDataLoader(
        hetero_graph, train_eids_dict, 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=16
        )

print("Length of dataloader is",len(train_dataloader))

valid_dataloader = dgl.dataloading.EdgeDataLoader(
        hetero_graph, valid_eids_dict, 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=16
        )

def train_single_epoch(train_dataloader):
    
    total_loss = 0
    batch_num = 0

    for _, pos_g, neg_g, blocks in train_dataloader:
        
        optimizer.zero_grad()

        input_features = blocks[0].srcdata['features']
        
        _, pos_score, neg_score = model(blocks,
                                        input_features,
                                        pos_g,
                                        neg_g)
        
        batch_num += 1
        # print(pos_score, neg_score)
        # print(batch_num%10000)
        if batch_num%10000 == 0:
            print(batch_num)

        loss = max_margin_loss(pos_score, neg_score)

#         optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss
    

model = MPNNConvModel(hetero_graph, dim_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0)

for i in range(epochs):
    
    print(f"Epoch {i}")

    pass
    
    training_loss = train_single_epoch(train_dataloader)
    
    print(f"Training Loss : {training_loss}")
    
#     validation_loss = validation_single_epoch(valid_dataloader)
    
#     print(f"Validation Loss : {validation_loss}")
    