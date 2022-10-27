import sys  
sys.path.insert(0, '/Users/pratikaher/DGL/GRecSy/')

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

df_rating = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'],  engine='python')
df_user = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/users.dat', sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],  engine='python')
df_movie = pd.read_csv('/Users/pratikaher/DGL/graph-rec/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genre'],  engine='python')

df_temp = df_rating.merge(df_movie, left_on='movie_id', right_on='movie_id', how='left')
df_final = df_temp.merge(df_user, left_on='user_id', right_on='user_id', how='left')

df_final = df_final[["user_id","movie_id","rating","age"]]

print(df_final.tail())

hetero_graph = dgl.heterograph({('user', 'rates', 'movie'): (df_final['user_id'].to_numpy(), df_final['movie_id'].to_numpy())})


