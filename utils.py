# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import numpy as np
import random
import torch
import dask.dataframe as dd


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    """
    smiles, graphs, labels = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels


def collate_movie_graphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    """

    import random
    labels = [0]*3020 + [1]*3020
    labels = random.shuffle(labels)

    graphs = data

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return bg, labels


def load_model(args):
    if args['model'] == 'SchNet':
        from dgllife.model import SchNetPredictor
        model = SchNetPredictor(node_feats=args['node_feats'],
                                hidden_feats=args['hidden_feats'],
                                predictor_hidden_feats=args['predictor_hidden_feats'],
                                n_tasks=args['n_tasks'])

    if args['model'] == 'MGCN':
        from dgllife.model import MGCNPredictor
        model = MGCNPredictor(feats=args['feats'],
                              n_layers=args['n_layers'],
                              predictor_hidden_feats=args['predictor_hidden_feats'],
                              n_tasks=args['n_tasks'])

    if args['model'] == 'MPNN':
        from model import MPNNPredictor
        model = MPNNPredictor(node_in_feats=args['node_in_feats'],
                              edge_in_feats=args['edge_in_feats'],
                              node_out_feats=args['node_out_feats'],
                              edge_hidden_feats=args['edge_hidden_feats'],
                              n_tasks=args['n_tasks'])

    return model

def _split_data(movielens):

        num_test_items = movielens['user_id'].nunique()//10

        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(
            ascending=False, method="first"
        )
        movielens_test = movielens[movielens["rating_order"] > num_test_items]
        movielens_train = movielens[movielens["rating_order"] <= num_test_items]
        return movielens_train, movielens_test

def train_test_split_by_time(df, timestamp, user):
    """Creates train-test splits of dataset by training on past to predict the future.

    This is the train-test split method most of the recommender system papers running on MovieLens
    takes. It essentially follows the intuition of "training on the past and predict the future".
    One can also change the threshold to make validation and test set take larger proportions.

    Args:
        df (pd.DataFrame): dataframe with user id's and timestamps
        timestamp (str): name of column with timestamp data
        user (str): name of column with user id data

    Returns:
        Train, validation, and test indices of edges, represented as NumPy arrays.
    """
    # Create masks for train, validation, and test sets
    df['train_mask'] = np.ones((len(df),), dtype=np.bool) # all true
    df['val_mask'] = np.zeros((len(df),), dtype=np.bool) # all false
    df['test_mask'] = np.zeros((len(df),), dtype=np.bool) # all false

    # Split dataframe into dask dataframe partitions
    df = dd.from_pandas(df, npartitions=10)

    def train_test_split(df):
        """Sorts dataset by timestamp and creates train, validation, and test mask columns.

        Args:
            df (pd.DataFrame): a dataframe with timestamp data

        Returns:
            A dataframe with train, validation, and test mask columns.
        """
        df = df.sort_values([timestamp]) # sort dataframe by timestamp

        # if more than 1 row, move last row from train mask to test mask
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True

        # if more than 2 rows, move 2nd to last row from train mask to validation mask
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    df = df.groupby(user, group_keys=False) \
           .apply(train_test_split) \
           .compute(scheduler='processes') \
           .sort_index()

    # print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))

    return df['train_mask'].to_numpy().nonzero()[0], \
           df['val_mask'].to_numpy().nonzero()[0], \
           df['test_mask'].to_numpy().nonzero()[0]
