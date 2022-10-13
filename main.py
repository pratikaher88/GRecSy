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

from torch.utils.data import DataLoader

from dgllife.utils import EarlyStopping, Meter
from model import compute_loss

def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

def regress(args, model, bg, node_subgraph_negative):
    bg = bg.to(args['device'])
    if args['model'] == 'MPNN':
        h = bg.ndata.pop('age')
        e = bg.edata.pop('rating')
        h, e = h.to(args['device'], dtype=torch.float), e.to(args['device'], dtype=torch.float)
        # print("object size", h.size(), e.size())
        return model(bg, h, e, node_subgraph_negative)
    elif args['model'] in ['SchNet', 'MGCN']:
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_types, edge_distances = node_types.to(args['device']), \
                                     edge_distances.to(args['device'])
        return model(bg, node_types, edge_distances)


def run_a_train_epoch_full(args, epoch, model, g, dataloader, optimizer):

    model.train()
    total_loss = 0

    batch_number  = 0
    for input_nodes, node_subgraph, node_subgraph_negative, blocks in dataloader:
        pos_score, neg_score = regress(args, model, node_subgraph, node_subgraph_negative)
        loss = compute_loss(pos_score, neg_score)

        batch_number += 1
        if batch_number % 100 == 0:
            print("Edge batch {}".format(batch_number))

        if epoch > 0:  # For the epoch 0, no training (just report loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    train_avg_loss = total_loss / batch_number
    
    return train_avg_loss



def run_a_train_epoch(args, epoch, model, g, node_subgraph_negative,
                      loss_criterion, optimizer):
    model.train()

    # print("Subgraph information", g ,g.num_nodes, g.num_edges, type(g))

    pos_score, neg_score = regress(args, model, g, node_subgraph_negative)

    loss = compute_loss(pos_score, neg_score)

    if epoch > 0:  # For the epoch 0, no training (just report loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(loss.item())

    # print(prediction)

    # print("FP DONE")

    # train_meter = Meter()
    # for batch_id, batch_data in enumerate(data_loader):
    #     print("Batch data:")
    #     smiles, bg, labels = batch_data
    #     batch_graph = dgl.block_to_graph(labels[0])
    #     print("Final step", bg, dgl.block_to_graph(labels[0]).edata)
    #     prediction = regress(args, model, batch_graph)
    #     pass

        # labels = labels.to(args['device'])
        # prediction = regress(args, model, bg)
        # loss = (loss_criterion(prediction, labels)).mean()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # train_meter.update(prediction, labels)
    
    # total_score = np.mean(train_meter.compute_metric(args['metric_name']))
    # print('epoch {:d}/{:d}, training {} {:.4f}'.format(
    #     epoch + 1, args['num_epochs'], args['metric_name'], total_score))

if __name__ == "__main__":

    import sys
    sys.argv=['']
    del sys

    parser = argparse.ArgumentParser(description='Alchemy for Quantum Chemistry')
    parser.add_argument('-m', '--model', type=str, choices=['MPNN', 'SchNet', 'MGCN'], default='MPNN',
                        help='Model to use')
    parser.add_argument('-n', '--num-epochs', type=int, default=250,
                        help='Maximum number of epochs for training')
    args = parser.parse_args().__dict__
    args['dataset'] = 'Alchemy'
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    df_rating = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'],  engine='python')
    df_user = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/users.dat', sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],  engine='python')
    df_movie = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genre'],  engine='python')

    df_temp = df_rating.merge(df_movie, left_on='movie_id', right_on='movie_id', how='left')
    df_final = df_temp.merge(df_user, left_on='user_id', right_on='user_id', how='left')

    print(df_user.head())


    # train, val, test = train_test_split_by_time(df_final, 'timestamp', 'user_id')

    movielens_train, movielens_valid = _split_data(df_final)

    print(movielens_valid.head())

    print(movielens_train.shape, movielens_valid.shape, movielens_valid['user_id'].nunique(), len(movielens_valid['age'].to_numpy()))

    train_graph = dgl.graph((movielens_train['user_id'].to_numpy(), movielens_train['movie_id'].to_numpy()))
    valid_graph = dgl.graph((movielens_valid['user_id'].to_numpy(), movielens_valid['movie_id'].to_numpy()))

    isolated_nodes = ((train_graph.in_degrees() == 0) & (train_graph.out_degrees() == 0)).nonzero().squeeze(1)
    train_graph = dgl.remove_nodes(train_graph, isolated_nodes)

    isolated_nodes = ((valid_graph.in_degrees() == 0) & (valid_graph.out_degrees() == 0)).nonzero().squeeze(1)
    valid_graph = dgl.remove_nodes(valid_graph, isolated_nodes)

    # g.edata['rating'] = torch.tensor(df_final['rating'].to_numpy())
    # g.ndata['age'] = torch.tensor(df_user['age'].to_numpy()).float() / 100

    print("Number of Nodes", len(valid_graph.nodes()))
    print("AGE length", torch.tensor(df_user['age'].to_numpy()).size())

    # train_graph.edata['rating'] = torch.unsqueeze(torch.tensor(movielens_train['rating'].to_numpy()), dim = 1)
    # train_graph.ndata['age'] = torch.unsqueeze(torch.tensor(movielens_train['age'].to_numpy()).float() / 100, dim = 1)

    valid_graph.edata['rating'] = torch.unsqueeze(torch.tensor(movielens_valid['rating'].to_numpy()), dim = 1)
    valid_graph.ndata['age'] = torch.unsqueeze(torch.tensor(movielens_valid['age'].to_numpy()).float() / 100, dim = 1)

    print("Graph Created")
    
    # train_loader = DataLoader(dataset=g,
    #                           batch_size=args['batch_size'],
    #                           shuffle=True,
    #                           collate_fn=collate_movie_graphs)

    # sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    # data_loader = dgl.dataloading.NodeDataLoader( g, g.nodes(), sampler,
    #                 batch_size=1024, shuffle=True, drop_last=False)
    
    # for i, mini_batch in enumerate(ddfata_loader):
    #     print(i, len(mini_batch))
    #     input_nodes, output_nodes, subgs = mini_batch
    #     print(len(input_nodes), len(output_nodes), subgs)

    model = load_model(args)
    print("Model Loaded")

    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    stopper = EarlyStopping(mode=args['mode'], patience=args['patience'])
    model.to(args['device'])

    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    sampler_n = dgl.dataloading.negative_sampler.Uniform(
        args['batch_size']
    )

    # data_loader = dgl.dataloading.NodeDataLoader( g, g.nodes(), sampler,
    #                 batch_size=15, shuffle=True, drop_last=False)

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_graph, train_graph.nodes(), 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=args['batch_size']
        )
    
    valid_dataloader = dgl.dataloading.EdgeDataLoader(
        valid_graph, valid_graph.nodes(), 
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
        shuffle=True, drop_last=False,
        batch_size=args['batch_size']
        )

    for epoch in range(10):

        average_loss = run_a_train_epoch_full(args, epoch, model, train_graph, dataloader, optimizer)

        print("Epoch :{} has a average loss of :{}".format(epoch, average_loss))
        
        print('VALIDATION LOSS')
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            i = 0
            for _, pos_g, neg_g, blocks in valid_dataloader:

                print(pos_g, neg_g, i)

                pos_score, neg_score = regress(args, model, pos_g, neg_g)
                loss = compute_loss(pos_score, neg_score)
                print(loss)
                total_val_loss += loss.item()
                i += 1
            
            val_avg_loss = total_val_loss / i
        
        print("This model has a average validation loss of :{}".format(val_avg_loss))
    
    # for i, mini_batch in enumerate(data_loader):
    #     # input_nodes, output_nodes, subgs = mini_batch
    #     print(dgl.block_to_graph(mini_batch[-1][0]))
    #     break

    # for epoch in range(10):
    #     # Train
    #     print("Epoch :", epoch)
    #     node_subgraph = dgl.node_subgraph(g, range(1,16))
    #     node_subgraph_negative = construct_negative_graph(node_subgraph, 5)

    #     print(node_subgraph, node_subgraph_negative)
        # run_a_train_epoch(args, epoch, model, node_subgraph, node_subgraph_negative, loss_fn, optimizer)
