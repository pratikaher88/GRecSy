"""MovieLens dataset"""
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import get_download_dir
from utils2 import to_etype_name

_urls = {
    'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-10m' : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
}

READ_DATASET_PATH = get_download_dir()
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

class MovieLens(object):

    def __init__(self, name, test_ratio=0.1, valid_ratio=0.1):
        self._name = name
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        # download and extract
        download_dir = get_download_dir()
        zip_file_path = '{}/{}.zip'.format(download_dir, name)
        # download(_urls[name], path=zip_file_path)
        # extract_archive(zip_file_path, '{}/{}'.format(download_dir, name))

        if name == 'ml-10m':
            root_folder = 'ml-10M100K'
        else:
            root_folder = name
        self._dir = os.path.join(download_dir, name, root_folder)
        
        print("Starting processing {} ...".format(self._name))

        self.user_info = self._load_raw_user_info()
        self.movie_info = self._load_raw_movie_info()
        self.all_rating_info = self._load_rating_info(os.path.join(self._dir, 'ratings.dat'), '::')

        num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
        shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
        self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
        self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]

        print('......')
        num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
        self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
        self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
        self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)

        print("All rating edges : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating edges : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating edges : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating edges : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating edges  : {}".format(self.test_rating_info.shape[0]))

        self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                 cmp_col_name="id",
                                                 reserved_ids_set=set(self.all_rating_info["user_id"].values),
                                                 label="user")
        
        self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                  cmp_col_name="id",
                                                  reserved_ids_set=set(self.all_rating_info["movie_id"].values),
                                                  label="movie")

        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}

        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)

        self.user_feature = th.FloatTensor(self._process_user_fea())
        self.movie_feature = th.FloatTensor(self._process_movie_fea())
    
        self.user_feature_shape = self.user_feature.shape
        self.movie_feature_shape = self.movie_feature.shape

        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=False)

        self.valid_enc_graph = self._generate_enc_graph(valid_rating_pairs, valid_rating_values, add_support=False)

        self.test_enc_graph = self._generate_enc_graph(test_rating_pairs, test_rating_values, add_support=False)

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            self._npairs(self.train_enc_graph)))
        # print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
        #     self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
            self._npairs(self.valid_enc_graph)))
        # print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
        #     self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
            self._npairs(self.test_enc_graph)))
        # print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
        #     self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
        #     self.test_dec_graph.number_of_edges()))

        
        self.train_enc_graph.nodes['user'].data['features'] = self.user_feature
        self.train_enc_graph.nodes['movie'].data['features'] = self.movie_feature
        train_rating_info = th.FloatTensor(self.train_rating_info['rating'].values.astype(np.float32))
        self.train_enc_graph.edges['rates'].data['rating'] =  train_rating_info
        self.train_enc_graph.edges['rev-rates'].data['rating'] =  train_rating_info

        self.valid_enc_graph.nodes['user'].data['features'] = self.user_feature
        self.valid_enc_graph.nodes['movie'].data['features'] = self.movie_feature
        valid_rating_info = th.FloatTensor(self.valid_rating_info['rating'].values.astype(np.float32))
        self.valid_enc_graph.edges['rates'].data['rating'] =  valid_rating_info
        self.valid_enc_graph.edges['rev-rates'].data['rating'] =  valid_rating_info

        self.test_enc_graph.nodes['user'].data['features'] = self.user_feature
        self.test_enc_graph.nodes['movie'].data['features'] = self.movie_feature
        test_rating_info = th.FloatTensor(self.test_rating_info['rating'].values.astype(np.float32))
        self.test_enc_graph.edges['rates'].data['rating'] =  test_rating_info
        self.test_enc_graph.edges['rev-rates'].data['rating'] =  test_rating_info

    def _npairs(self, graph):
        return sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):

        # print("Rating information",rating_pairs, rating_values)

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs

        data_dict.update({
                ('user', 'rates', 'movie'): (rating_row, rating_col),
                ('movie', 'rev-rates', 'user'): (rating_col, rating_row)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2
 
        return graph

    # def _generate_dec_graph(self, rating_pairs):
    #     ones = np.ones_like(rating_pairs[0])
    #     user_movie_ratings_coo = sp.coo_matrix(
    #         (ones, rating_pairs),
    #         shape=(self.num_user, self.num_movie), dtype=np.float32)
    #     g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
    #     return dgl.heterograph({('user', 'rate', 'movie'): g.edges()}, 
    #                            num_nodes_dict={'user': self.num_user, 'movie': self.num_movie})

    # @property
    # def num_links(self):
    #     return self.possible_rating_values.size

    # @property
    # def num_user(self):
    #     return self._num_user

    # @property
    # def num_movie(self):
    #     return self._num_movie

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):

        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _load_rating_info(self, file_path, sep):

        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_info

    def _load_raw_user_info(self):

        user_info = pd.read_csv(os.path.join(self._dir, 'users.dat'), sep='::', header=None,
                                names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        
        return user_info

    def _process_edge_fea(self):
        pass



    def _process_user_fea(self):

        ages = self.user_info['age'].values.astype(np.float32)
        gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
        all_occupations = set(self.user_info['occupation'])
        occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
        occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                        dtype=np.float32)
        occupation_one_hot[np.arange(self.user_info.shape[0]),
                            np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
        user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                        gender.reshape((self.user_info.shape[0], 1)),
                                        occupation_one_hot], axis=1)

        return user_features

    def _load_raw_movie_info(self):

        GENRES = GENRES_ML_1M

        file_path = os.path.join(self._dir, 'movies.dat')
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

    def _process_movie_fea(self):

        import torchtext
        from torchtext.data.utils import get_tokenizer

        GENRES = GENRES_ML_1M

        # Old torchtext-legacy API commented below
        # TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm') # new API (torchtext 0.9+)
        embedding = torchtext.vocab.GloVe(name='840B', dim=300)

        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                print('{} cannot be matched, index={}, name={}'.format(title, i, self._name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()

            title_embedding[i, :] = embedding.get_vecs_by_tokens(tokenizer(title_context)).numpy().mean(axis=0)
            release_years[i] = float(year)
        movie_features = np.concatenate((title_embedding,
                                         (release_years - 1950.0) / 100.0,
                                         self.movie_info[GENRES]),
                                        axis=1)
        return movie_features

if __name__ == '__main__':
    MovieLens("ml-100k", device=th.device('cpu'), symm=True)