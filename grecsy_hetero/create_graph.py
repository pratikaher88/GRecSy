from dgl.data.utils import get_download_dir
import os
import torch
import pandas as pd
import numpy as np
import dgl

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]

def create_graph():

    download_dir = get_download_dir()
    _dir = os.path.join(download_dir, "ml-1m", "ml-1m")

    user_data = _load_raw_user_info(os.path.join(_dir, 'users.dat'))
    movie_data = _load_raw_movie_info(os.path.join(_dir, 'movies.dat'))
    ratings_data = _load_rating_info(os.path.join(_dir, 'ratings.dat'), '::')


    movie_data["movie_id"] = movie_data.index
    user_data["user_id"] = user_data.index

    HM_movie_ID = dict(zip(movie_data.id,movie_data.movie_id))
    HM_user_ID = dict(zip(user_data.id,user_data.user_id))

    ratings_data["user_id"] = ratings_data["user_id"].map(HM_user_ID)
    ratings_data["movie_id"] = ratings_data["movie_id"].map(HM_movie_ID)

    graph_data = {
        ('user','rates','movie') : (ratings_data['user_id'].to_numpy(), ratings_data['movie_id'].to_numpy()),
        ('movie','rev-rates','user') : (ratings_data['movie_id'].to_numpy(), ratings_data['user_id'].to_numpy())
    }

    movie_hetero_graph = dgl.heterograph(graph_data)

    movie_hetero_graph.nodes['movie'].data['node_ID'] = torch.tensor(movie_data["movie_id"]).squeeze()
    movie_hetero_graph.nodes['user'].data['node_ID'] = torch.tensor(user_data["user_id"]).squeeze()

    isolated_nodes =  (movie_hetero_graph.in_degrees(etype='rates') == 0).nonzero().squeeze(1)

    movie_hetero_graph.remove_nodes(isolated_nodes.clone().detach(), ntype='movie')

    movieid_to_feat = _process_movie_fea(movie_data)
    userid_to_feat = _process_user_fea(user_data)

    mapped_movie_features = []
    for value in movie_hetero_graph.nodes['movie'].data['node_ID'].tolist():
        mapped_movie_features.append(movieid_to_feat[value])

    mapped_user_features = []
    for value in movie_hetero_graph.nodes['user'].data['node_ID'].tolist():
        mapped_user_features.append(userid_to_feat[value])
    
    movie_hetero_graph.nodes['user'].data['features'] = torch.stack(mapped_user_features, axis=0)
    movie_hetero_graph.nodes['movie'].data['features'] = torch.stack(mapped_movie_features, axis=0)

    return movie_hetero_graph

def _process_user_fea(user_data):
    HM = {}
    
    for index, row in user_data.iterrows():
        
        age = row['age']
        gender = (row['gender'] == 'F')
        
        HM[row['user_id']] = torch.FloatTensor([age, gender])
        
    return HM

def _process_movie_fea(movie_data):
    
    import re
    
    HM = {}
    p = re.compile(r'(.+)\s*\((\d+)\)')
    
    for index, row in movie_data.iterrows():
        match_res = p.match(row['title'])
        
        if match_res is None:
            print('{} cannot be matched, index={}'.format(row['title'], index))
            title_context, year = row['title'], 1950
        else:
            title_context, year = match_res.groups()
        
        HM[row['movie_id']] = torch.FloatTensor([ (float(year)- 1950.0) / 100.0])

    return HM

def _load_rating_info(file_path, sep):

        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_info

def _load_raw_user_info(file_path):

    user_info = pd.read_csv(file_path, sep='::', header=None,
                            names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')

    return user_info
    
def _load_raw_movie_info(file_path):

        GENRES = GENRES_ML_1M

        movie_info = pd.read_csv(file_path, sep='::', header=None,
                                    names=['id', 'title', 'genres'], encoding='iso-8859-1')
        genre_map = {ele: i for i, ele in enumerate(GENRES)}
        genre_map['Children\'s'] = genre_map['Children']
        genre_map['Childrens'] = genre_map['Children']
        movie_genres = np.zeros(shape=(movie_info.shape[0], len(GENRES)), dtype=np.float32)
        for i, genres in enumerate(movie_info['genres']):
            for ele in genres.split('|'):
                if ele in genre_map:
                    movie_genres[i, genre_map[ele]] = 1.0
                else:
                    print('genres not found, filled with unknown: {}'.format(genres))
                    movie_genres[i, genre_map['unknown']] = 1.0
        
        for idx, genre_name in enumerate(GENRES):
            assert idx == genre_map[genre_name]
            movie_info[genre_name] = movie_genres[:, idx]
        
        movie_info.drop(columns=["genres"])

        return movie_info