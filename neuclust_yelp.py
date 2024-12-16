
import pickle
with open('business_data.pkl', 'rb') as file: 

    # Call load method to deserialze 
    business_data = pickle.load(file)

with open('review_data.pkl', 'rb') as file: 

    # Call load method to deserialze 
    review_data = pickle.load(file)

with open('user_data.pkl', 'rb') as file: 

    # Call load method to deserialze 
    user_data = pickle.load(file)

with open('X_prefs_clusters_yelp_20.pkl', 'rb') as file: 

    # Call load method to deserialze 
    X_prefs_clusters = pickle.load(file)

dict_categories_1 = []
for ind in business_data.index:
    for el in business_data['categories'][ind].split(", "):
        if el=='Restaurants':
            dict_categories_1.append(business_data['business_id'][ind])
        
filtered_business_df_1 = business_data[business_data['business_id'].isin(dict_categories_1)]
dict_businesses_1 = []
for ind in filtered_business_df_1.index:
    dict_businesses_1.append(filtered_business_df_1['business_id'][ind])
dict_buss_id = dict_categories_1

dict_new_categories_1 = []
for ind in filtered_business_df_1.index:
    cats = filtered_business_df_1['categories'][ind]
    cats = cats.split(", ")
    for el in cats:
        if el not in dict_categories_1:
            dict_new_categories_1.append(el)

filtered_review_df = review_data[review_data['business_id'].isin(filtered_business_df_1.business_id)]
filtered_review_df = filtered_review_df.convert_dtypes()
filtered_user_ids=filtered_review_df.user_id.unique()
corpus = []
for ind in filtered_business_df_1.index:
    cats = filtered_business_df_1['categories'][ind]
    cats = cats.split(", ")
    strr = cats[0]
    for i in range(len(cats)-1):
        strr = strr + ' ' + cats[i+1]
    corpus.append(strr)

corpus_final = []
for doc in corpus:
    if doc.__contains__('&') == True:
        new_string = doc.replace("&", "")   
        new_string = new_string.replace("  ", " ")
        corpus_final.append(new_string)
    else:
        corpus_final.append(doc)
        
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
tr_idf_model  = TfidfVectorizer()
tf_idf_vector = tr_idf_model.fit_transform(corpus_final)
tf_idf_array = tf_idf_vector.toarray()
words_set = tr_idf_model.get_feature_names_out()
#words_set = tr_idf_model.get_feature_names()
df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
df_sum_columns = df_tf_idf.sum()
df_sum_filtered = df_sum_columns.loc[lambda x: x>=50]
filtered_columns = df_sum_filtered.index
filtered_columns_list =[]
for i in range(len(filtered_columns)):
    filtered_columns_list.append(filtered_columns[i])
df_filtered_tf_idf = df_tf_idf.loc[:,filtered_columns_list]
tf_idf_filtered_array = df_filtered_tf_idf.to_numpy()
from sklearn.cluster import KMeans
import numpy as np
# k_values_to_try = np.arange(2,40)
# for n_clusters in k_values_to_try:

#     kmeans = KMeans(n_clusters=n_clusters,
#                     random_state=0,
#                     )
#     labels_clusters = kmeans.fit_predict(tf_idf_filtered_array)
#     for n in range(n_clusters):
#         n_indices = np.where(labels_clusters == n)[0]

from sklearn.cluster import KMeans
# wcss = []
# for i in range(1,41):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#     kmeans.fit(tf_idf_filtered_array)
#     wcss.append(kmeans.inertia_)

# import matplotlib.pyplot as plt
# plt.plot(range(1,41), wcss)
# plt.title('The elbow method')
# plt.xlabel('The number of clusters')
# plt.ylabel('WCSS')
# plt.show()

n_clusters = 25

# kmeans = KMeans(n_clusters=n_clusters,
#                     init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0
#                     )
# labels_clusters = kmeans.fit_predict(tf_idf_filtered_array)
# X_prefs_clusters = []
# for n in range(n_clusters):
#     #n_indices = np.where(labels_clusters == n)[0]
#     #print(len(n_indices))
#     X_prefs_clusters.append(tf_idf_filtered_array[np.where(labels_clusters == n)[0],:])
import time
start_time = time.time()

class Reward:
    def __init__(self,arm, quality):
        self.quality = quality
        self.context = arm.context
        self.arm = arm
from abc import ABC



class Arm:
    def __init__(self, unique_id, context,true_mean):
        self.true_mean = true_mean  # Only used by the benchmark
        self.unique_id = unique_id
        self.context = context
"""
This abstract class represents a problem model that the ACC-UCB algorithm will run on.
"""

class ProblemModel(ABC):
    def __init__(self, num_rounds):
        self.num_workers = None  # must be set in a subclass
        self.num_rounds = num_rounds

    def get_available_arms(self, t):
        pass

    def oracle(self, K, g_list, t=None):
        pass

    def play_arms(self, t, slate):
        pass

    def reward_fun(self, outcome_arr, t=None):
        pass

    def get_regret(self, t, budget, slate):
        pass

    def get_task_budget(self, t):
        pass

    def get_total_reward(self, rewards, t=None):
        pass

import os

import h5py
import pickle
import random
random.seed(98765)
from typing import List
#from pytim import PyTimGraph
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

#import fs_loader
#from fs_loader import city_name
#from ProblemModel import ProblemModel
#from Reward import Reward
#from Arm import Arm

"""
Dynamic probabilistic maximum coverage problem model.
"""

SAVED_FILE_NAME = "movielens_simulation.hdf5"  # file where the saved simulation-ready hdf5 dataset will be written to
TEMP_TIM_PATH = "temp_graphs/"
TIM_EPSILON = 0.1

# def context_to_mean_fun(context):
#     """
#     context[0] = task location
#     context[1] = worker location
#     context[2] = task context
#     context[3] = worker context
#     """
#     return norm.pdf(np.linalg.norm(context[0] - context[1]), loc=0, scale=0.25) * \
#            (context[3] + context[2]) / 2 / norm.pdf(0, loc=0, scale=0.25)


def context_to_mean_fun(context):
    """
    """
    count = np.count_nonzero(context)
    if count == 0:
        count = 1
    return 2 / (1 + np.exp(-np.sum(context)/(2*count))) - 1


class MovielensProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, use_saved, budget,
                  dataset_path="datasets/ml_25m",
                  seed=98765, tim_graph_name="", scale_contexts=False,
                  ):
        global user_ids
        global X_prefs_clusters
        super().__init__(num_rounds)
        self.X_prefs_clusters = X_prefs_clusters
        self.num_rounds = num_rounds
        #self.saved_file_name = saved_file_name
        #self.edge_retrain_percentage = edge_retrain_percentage
        self.scale_contexts = scale_contexts
        self.tim_graph_name = tim_graph_name
        self.dataset_path = dataset_path
        #self.exp_left_nodes = exp_left_nodes
        #self.exp_right_nodes = exp_right_nodes
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        #if not use_saved:
        #    self.initialize_dataset()
        self.h5_file = None
        self.benchmark_superarm_list = None

    def set_benchmark_superarm_list(self, superarm_list):
        self.benchmark_superarm_list = superarm_list

    def get_available_arms(self, context):
        #if self.h5_file is None:
        #    self.h5_file = h5py.File(self.saved_file_name, "r")

        #t = t - 1
        # Construct a list of Arm objects (i.e., available edges)
        #context_dataset = self.h5_file[f"{t}"]["context_dataset"]
        #exp_outcome_dataset = self.h5_file[f"{t}"]["mean_dataset"]

        final_arm_list = []
        final_exp_out_list = []
        #length = len(user_ids)
        cnt = 0
        exp_out_list_all = []
        expp_out_list_all = []
        # for n in range(len(self.X_prefs_clusters)):
        #     random_chosen_arms_indices = np.random.choice(np.arange(len(X_prefs_clusters[n])),self.budget)
        #     context_random_chosen_arms = context[n][random_chosen_arms_indices]
        #     exp_out_list =np.zeros(self.budget)
        #     #arm_list = np.zeros(self.budget)
        #     arm_list = []
        #    #exp_out_list_all = exp_out_list_all + list(context[n])
            
        #     for j in range(len(context[n])):
        #         context_arm = context[n][j]
        #         expp = context_to_mean_fun(context_arm)
        #         expp_out_list_all.append(expp)
        #     #exp_out_list_all.append(expp_out_list)
        #     for i in range(self.budget):
        #         context_arm = context_random_chosen_arms[i,:]
        #         exp_outcome = context_to_mean_fun(context_arm)
        #         exp_out_list[i] = exp_outcome
        #         #arm_list[i] = Arm(cnt, np.array(context_arm), exp_outcome)
        #         arm_list.append(Arm(cnt, np.array(context_arm), exp_outcome))
        #         #arm_list.append(Arm(len(arm_list), np.array(context[:,i]), exp_outcome))
        #         #exp_out_list.append(exp_outcome)
        #         cnt = cnt+1
        #     final_arm_list.append(np.array(arm_list))
        #     final_exp_out_list.append(exp_out_list)
            
            
        for n in range(len(self.X_prefs_clusters)):
            #random_chosen_arms_indices = np.random.choice(np.arange(len(X_prefs_clusters[n])),self.budget)
            context_cluster = context[n]
            exp_out_list = np.zeros(np.shape(context_cluster)[0])
            #arm_list = np.zeros(self.budget)
            arm_list = []
            #exp_out_list_all = exp_out_list_all + list(context[n])
            
            for j in range(len(context[n])):
                context_arm = context[n][j]
                expp = context_to_mean_fun(context_arm)
                expp_out_list_all.append(expp)
            #exp_out_list_all.append(expp_out_list)
            for i in range(len(exp_out_list)):
                context_arm = context_cluster[i,:]
                exp_outcome = context_to_mean_fun(context_arm)
                exp_out_list[i] = exp_outcome
                #arm_list[i] = Arm(cnt, np.array(context_arm), exp_outcome)
                arm_list.append(Arm(cnt, np.array(context_arm), exp_outcome))
                #arm_list.append(Arm(len(arm_list), np.array(context[:,i]), exp_outcome))
                #exp_out_list.append(exp_outcome)
                cnt = cnt+1
            final_arm_list.append(np.array(arm_list))
            final_exp_out_list.append(exp_out_list)
        #print(len(expp_out_list_all))
        return final_arm_list,final_exp_out_list,expp_out_list_all

    def get_regret(self, exp_out_list,budget,slate,available_arms):
        #df = self.df.loc[t]
        ld = sorted(exp_out_list,reverse=True)
        highest_means = ld[:budget]
        algo_mean_prod = 0
        bench_mean_prod = 0
        for arm in slate:
            algo_mean_prod += available_arms[arm.unique_id].true_mean
        for mean in highest_means:
            bench_mean_prod += mean

        return bench_mean_prod - algo_mean_prod

    def get_optimal_super_arm_reward(self,exp_out_list,budget):
        ld = sorted(exp_out_list,reverse=True)
        highest_means = ld[:budget]
        #print(highest_means)
        reward_list = [1.0 * np.random.binomial(1, mean) for mean in highest_means]
        reward_sum = 0
        for reward in reward_list:
            reward_sum += reward
        if reward_sum >= len(reward_list)*8/10:
            return 1
        return 0
        #return reward_sum/len(reward_list)

    def get_total_reward(self, rewards):
        # if self.h5_file is None:
        #     self.h5_file = h5py.File(self.saved_file_name, "r")

        # t = t - 1  # because time starts from 1 but index starts from 0
        # go through edges and trigger
        # activated_users = set()  # activated right nodes
        # for reward in rewards:
        #     edge_idx = reward.worker.unique_id

        #     # get user id corresponding to this edge
        #     user_id = self.h5_file[f"{t}"]['edge_dataset'][edge_idx][1]
        #     is_triggered = reward.performance
        #     if is_triggered == 1:
        #         activated_users.add(user_id)
        # return len(activated_users)

        reward_sum = 0
        for reward in rewards:
            reward_sum += reward.quality  # Total reward is lin sum
        if reward_sum >= len(rewards)*8/10:
            return 1
        return 0
        # return reward_sum/len(rewards)

    def play_arms(self, slate):
        reward_list = [Reward(arm, 1.0 * np.random.binomial(1, arm.true_mean)) for arm in slate]
        rew_list = []
        for arm in slate:
            rew_list.append(arm.true_mean)
        #print(rew_list)
        return reward_list

    def sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def oracle(self, budget, g_list):
        return np.argsort(g_list)[-budget:]

    def get_task_budget(self, t):
        return self.budget

    def initialize_dataset(self):
        print("Generating dataset from MovieLens...")

        # sample number of left and right nodes for all rounds. Needed to allocate h5 dataset
        #left_node_count_arr = self.rng.poisson(self.exp_left_nodes, self.num_rounds)
        #right_node_count_arr = self.rng.poisson(self.exp_right_nodes, self.num_rounds)

        # create h5 dataset
        #h5_file = h5py.File(self.saved_file_name, "w")

        # load movielens dataset
        movies_df = pd.read_csv("movies_all.csv")
        ratings_df = pd.read_csv("ratings_all.csv")

        # remove ratings older than Jan 2015
        ratings_df = ratings_df[ratings_df.timestamp > 1420070400]

        genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western', 'IMAX', '(no genres listed)']

        # remove movies with fewer than 200 ratings
        user_id_num_ratings = ratings_df.userId.value_counts()
        filtered_ratings = ratings_df[ratings_df.userId.isin(user_id_num_ratings.index[user_id_num_ratings.ge(200)])]
        filtered_movie_ids = filtered_ratings.movieId.unique()

        # make movie df index movie ID
        filtered_movie_df = movies_df.set_index("movieId").loc[filtered_movie_ids]


        movie_id_genre_dict = np.zeros((20,len(filtered_movie_ids)))
        user_id_prefs_dict = dict()
        f = 0
        for movie_id in filtered_movie_ids:
            #if movie_id not in movie_id_genre_dict:
            movie_id_genre_dict[:,f] = np.array(
                        [1 if g in filtered_movie_df.genres.loc[movie_id] else 0 for g in genres])
            f = f+1
        user_ids = filtered_ratings.userId.unique()
        for user_id in user_ids:
            if user_id not in user_id_prefs_dict:
                user_pref_vec = np.zeros(len(genres))
                #movieid_rating_array = filtered_ratings[filtered_ratings.userId == user_id]
                count = np.zeros(20)
                for j in range(len(filtered_ratings[filtered_ratings.userId == user_id][["movieId","rating"]].values)):
                    movie_idd = (filtered_ratings[filtered_ratings.userId == user_id]
                                          [["movieId","rating"]].values[j][0])
                    #print(movie_id_index)
                    movie_id_index = np.where(filtered_movie_ids == movie_idd)[0]
                    rating = filtered_ratings[filtered_ratings.userId == user_id][["movieId","rating"]].values[j][1]
                    genre_vec_of_mov = movie_id_genre_dict[:,movie_id_index]
                    multiplied = rating*np.transpose(genre_vec_of_mov)
                    multiplied = np.reshape(multiplied,20)
                    user_pref_vec = user_pref_vec + multiplied
                    count = count + np.reshape(genre_vec_of_mov,20)

                for i in range(20):
                    if count[i] == 0:
                        user_pref_vec[i] = 0
                    else:
                        user_pref_vec[i] = user_pref_vec[i]/count[i]
                user_id_prefs_dict[user_id] = user_pref_vec

        return movie_id_genre_dict,user_id_prefs_dict,user_ids,filtered_movie_ids

problem_model = MovielensProblemModel(1000, False, 5)

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np



class Bandit_multi:
    def __init__(self, name, is_shuffle=False, seed=None):
        # Fetch data
        # global df
        # global movie_id_genre_dict
        # global user_id_prefs_dict
        # global filtered_movie_ids
        global filtered_user_ids
        global X_prefs_clusters
        global filtered_review_df
        global filtered_business_df_1
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'heart':
            X, y = fetch_openml(data_id = 43682, return_X_y=True,parser ='auto')
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        # elif name == 'gowalla':
        #     X = df
        # elif name == 'movielens':
        #     X_movie = movie_id_genre_dict
        #     self.X_user = user_id_prefs_dict
        # else:
        #     raise RuntimeError('Dataset does not exist')
        # Shuffle data
        # if is_shuffle:
        #     self.X= shuffle(X, random_state=seed)
        #     df = self.X
        # else:
        #     #self.X, self.y = X, y
        #     #self.X = X
        #     if name == "movielens":
        #         self.X_movie = X_movie
        #         self.X_user = user_id_prefs_dict
        # generate one_hot coding:
        #self.y_arm = OrdinalEncoder(
        #    dtype=np.int).fit_transform(self.y.values.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = 1000
        #self.n_arm = np.max(self.y_arm) + 1
        #self.dim = self.X.shape[1] * self.n_arm
        self.dim = np.shape(X_prefs_clusters[0])[1]
        self.X_prefs_clusters = X_prefs_clusters
        #self.act_dim = self.X.shape[1]

        #rng = np.random.default_rng()
        #movies = movie_id_genre_dict
        #rng.shuffle(movies,axis = 1)
        #self.movie_select_list = movies
        
    def step(self,t):
        #assert self.cursor < self.size
        #X_movie = np.zeros((20, 1))
        #selected_movie =random.choice(self.movie_select_list)
        #array_movie_select_list = np.array(self.movie_select_list)
        #indexx = np.where(array_movie_select_list == selected_movie)
        #self.movie_select_list.pop(indexx)
        user_id_t = t
        user_t = filtered_user_ids[user_id_t]
        aa = filtered_review_df[filtered_review_df['user_id'].str.contains(user_t,case = False)]
        business_ids_index = aa.business_id.index
        user_prefs_list = np.zeros(np.shape(tf_idf_filtered_array)[1])
        for i in range(aa.user_id.count()):
            bus_id_i = aa.business_id[business_ids_index[i]]
            bus_df_i = filtered_business_df_1[filtered_business_df_1['business_id'].str.contains(bus_id_i,case = False)]
            #bus_index_cnt = len(bus_df_i.index)
            #for i in range(bus_index_cnt):
            ind = bus_df_i.index[0]
            categoriess = filtered_business_df_1['categories'][ind]
            categoriess = categoriess.split(", ")
            cc = aa[aa['business_id'].str.contains(bus_id_i,case=False)]
            rating=(cc.to_numpy()[0][3])
            for el in categoriess:
                el = el.lower()
                if el.__contains__('&') == True:
                    new_string = el.replace("&", "")   
                    new_string = new_string.replace("  ", " ")    
                if el in filtered_columns_list:
                    indd = filtered_columns_list.index(el)
                    user_prefs_list[indd] += rating
        user_prefs_list = user_prefs_list/aa.user_id.count()
            
        #X_movie = self.movie_select_list[:,movie_id_t]
        #X_restaurant = X_prefs_clusters
        #X_user = self.X_prefs_clusters
        X_final = []
        #X = np.zeros((20,len(user_ids)))
        #for j in range(len(user_ids)):
        #    X[:,j] = X_user[user_ids[j]]*X_movie

        for j in range(len(X_prefs_clusters)):
            aaa = np.zeros((len(X_prefs_clusters[j]),np.shape(tf_idf_filtered_array)[1]))
            for i in range(len(X_prefs_clusters[j])):
                aaa[i] = self.X_prefs_clusters[j][i]*user_prefs_list
            X_final.append(aaa)
        return X_final        

import torch
import torch.nn as nn
#import torchmetrics
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
#from pytorch_lightning import LightningModule
from torch.optim import Adam, SGD
import math


class InterpretableLayer(nn.Module):
    __constants__ = ['in_features']
    in_features: int
    out_features: int
    weight: torch

    def __init__(self, in_features: int) -> None:
        super(InterpretableLayer, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.softsign = nn.Softsign()
        self.reset_parameters()
        self.relu = nn.ReLU()

    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=0)

    def forward(self, input: torch) -> torch:
        #  return input*torch.exp(self.weight) + self.bias  # DONE: take exp away an bias and add softsign
        return input * self.relu(self.weight)


class MonotonicLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch

    def pos_tanh(self, x):
        return torch.tanh(x) + 1.

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fn: str = 'exp') -> None:
        super(MonotonicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.fn = fn
        if fn == 'exp':
            self.pos_fn = torch.exp
        elif fn == 'square':
            self.pos_fn = torch.square
        elif fn == 'abs':
            self.pos_fn = torch.abs
        elif fn == 'sigmoid':
            self.pos_fn = torch.sigmoid
        else:
            self.fn = 'tanh_p1'
            self.pos_fn = self.pos_tanh
        self.reset_parameters()

    def reset_parameters(self) -> None:
        n_in = self.in_features
        if self.fn == 'exp':
            mean = math.log(1./n_in)
        else:
            mean = 0
        init.normal_(self.weight, mean=mean)
        if self.bias is not None:
            init.uniform_(self.bias, -1./n_in, 1./n_in)

    def forward(self, input: torch) -> torch:
        ret = torch.matmul(input, torch.transpose(self.pos_fn(self.weight), 0, 1))
        if self.bias is not None:
            ret = ret + self.bias
        return ret


class MonotonicConv2d(nn.Module):
    def pos_tanh(self, x):
        return torch.tanh(x) + 1.

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, fn='tanh_p1'):
        super(MonotonicConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self._filters = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.fn = fn
        if fn == 'exp':
            self.pos_fn = torch.exp
        elif fn == 'square':
            self.pos_fn = torch.square
        elif fn == 'abs':
            self.pos_fn = torch.abs
        elif fn == 'sigmoid':
            self.pos_fn = torch.sigmoid
        else:
            self.fn = 'tanh_p1'
            self.pos_fn = self.pos_tanh
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        filters = self.pos_fn(self._filters)
        return F.conv2d(x, filters, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,
                        groups=self.groups)


    def _step(self, batch, batch_idx, step_name):
        x, y = batch
        y = torch.flatten(y)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log(f'{step_name}/accuracy', accuracy, prog_bar=True)
        return {
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        self.train()
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.eval()
        ret = self._step(batch, batch_idx, 'valid')
        self.train()
        return ret

    def test_step(self, batch, batch_idx):
        self.eval()
        ret = self._step(batch, batch_idx, 'test')
        self.train()
        return ret


import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad

class Network(nn.Module):
    def __init__(self, dim = np.shape(X_prefs_clusters[0])[1], hidden_size=100):
        super(Network, self).__init__()
        #nn.Module.__init__(self)
        #self.fc1 = nn.Linear(dim, hidden_size)
        #self.activate = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_size, 1)
        self.model = nn.Sequential(nn.Linear(dim, hidden_size),nn.ReLU(),nn.Linear(hidden_size, 1))
    #def forward(self, x):

    #    return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiag:
    def __init__(self, style, K, dim, num_users, lamdba=1, nu=1, hidden=100):
        self.network = Network(dim, hidden_size=hidden)
        self.func = nn.Sequential(*self.network.model).cuda()
        self.func = extend(self.func)
        # self.func = Network(dim, hidden_size=hidden)
        self.context_list = None
        self.reward = None
        self.lamdba = lamdba
        #for p in self.func.parameters():
          #print(p.size())
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        # self.U = lamdba * torch.ones((self.total_param,))
        # self.U_random = lamdba * torch.ones((self.total_param,))
        self.U_random = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.num_rounds = 1000
        self.problem_model = MovielensProblemModel(self.num_rounds, use_saved = False, budget = 5)
        self.K = K
        self.num_users = num_users
        self.style = style
        #self.loss_func = extend(nn.MSELoss().cuda())
        self.loss_func = extend(nn.MSELoss())
        self.len = 0

    def predict(self, x):
        # Ensure x is a CUDA tensor
        x = x.to(device='cuda')
        return self.func(x)
    
    def get_ucb_cluster(self, contexts_tensor):
        # No need for the conversion from numpy as contexts_tensor is already a CUDA tensor
        contexts_tensor.requires_grad_(True)
    
        # Forward pass through the extended model
        mu = self.func(contexts_tensor)
    
        # Initialize the container for the gradients
        g_list = []
    
        # Use BackPACK to compute the gradients
        with backpack(BatchGrad()):
            # Compute the sum of mu over the batch dimension
            sum_mu = mu.sum(dim=1)
            sum_mu.backward(torch.ones_like(sum_mu))  # Compute per-sample gradients
        
        # Collect the per-sample gradients
        for p in self.func.parameters():
            if hasattr(p, 'grad_batch'):
                g = p.grad_batch.flatten(start_dim=1).detach()
                g_list.append(g)
            else:
                # Handle the case where grad_batch is not present
                print(f'grad_batch not found in parameter {p}')
    
        # Convert the list to a tensor
        g_list = torch.cat(g_list, dim=1)
    
        # Compute sigma for each item in the batch
        sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
    
        # Compute the sample return for each item in the batch
        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), 0.1 * sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + 0.01 * sigma.view(-1)
    
        return sample_r, g_list

    def update_params(self,g_list):
        #print(g_list.size())
        for g in g_list:
            self.U += g * g
        return 0


    def train(self, context, reward):
        # self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        # self.reward.append(reward)
        # optimizer = optim.SGD(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba)
        # length = len(self.reward)
        # index = np.arange(length)
        # np.random.shuffle(index)
        # cnt = 0
        # tot_loss = 0
        # while True:
        #     batch_loss = 0
        #     for idx in index:
        #         c = self.context_list[idx]
        #         r = self.reward[idx]
        #         optimizer.zero_grad()
        #         delta = self.func(c.cuda()) - r
        #         # delta = self.func(c) - r
        #         loss = delta * delta
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss += loss.item()
        #         tot_loss += loss.item()
        #         cnt += 1
        #         if cnt >= 1:
        #             return tot_loss / 1
        #     if batch_loss / length <= 1e-3:
        #         return batch_loss / length

        self.len += 1
        optimizer = optim.SGD(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba / self.len)
        if self.context_list is None:
            self.context_list = torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.reward = torch.tensor([reward], device='cuda', dtype=torch.float32)
        else:
            self.context_list = torch.cat((self.context_list, torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.reward = torch.cat((self.reward, torch.tensor([reward], device='cuda', dtype=torch.float32)))
        #if self.len % self.delay != 0:
        #    return 0
        for _ in range(40):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(self.context_list).view(-1)
            loss = self.loss_func(pred, self.reward)
            loss.backward()
            optimizer.step()
        return 0


import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim

class NetworkMono(nn.Module):
    def __init__(self, dim = 5, hidden_size=25):
        #nn.Module.__init__(self)
        super(NetworkMono, self).__init__()
        # self.pre_monotonic = InterpretableLayer(dim)
        # self.pre_monotonic.weight.data.normal_(0.0, 1 / sqrt(nb_neuron_inter_layer))

        self.model_super = nn.Sequential(
              MonotonicLayer(dim, hidden_size, fn='sigmoid'),
              nn.ReLU(),
              MonotonicLayer(hidden_size, 1, fn='sigmoid'),
        )

        # self.output = InterpretableLayer(1)
        #self.fc1 = nn.Linear(dim, hidden_size)
        #self.activate = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_size, 1)
        #self.model_super = nn.Sequential(nn.Linear(dim, hidden_size),nn.ReLU(),nn.Linear(hidden_size, 1))
    #def forward(self, x):
        #x = self.high_level_feats(x)
        # x = self.pre_monotonic(x)
        # x = self.monotonic(x)
        # x = self.output(x)
        # return x
        #return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiagMono:
    def __init__(self, dim=5, lamdba=1, nu=1, hidden=25):
        self.func_super = extend(NetworkMono(dim, hidden_size=hidden).model_super.cuda())
        # self.func_super = NetworkMono(dim, hidden_size=hidden)
        self.rewards_list = None
        self.super_reward = None
        self.len = 0
        self.lamdba = lamdba
        #for p in self.func.parameters():
          #print(p.size())
        self.total_param = sum(p.numel() for p in self.func_super.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        # self.U = lamdba * torch.ones((self.total_param,))
        self.nu = nu
        self.num_rounds = 1000
        self.problem_model = MovielensProblemModel(self.num_rounds,budget = 5, use_saved = False)
        #self.loss_func_super = extend(nn.MSELoss().cuda())
        self.loss_func_super = extend(nn.MSELoss())

    def train(self, rewards, super_reward):
        # self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        # self.super_reward.append(reward)
        # optimizer = optim.SGD(self.func_super.parameters(), lr=1e-3, weight_decay=self.lamdba)
        # length = len(self.super_reward)
        # index = np.arange(length)
        # np.random.shuffle(index)
        # cnt = 0
        # tot_loss = 0
        # while True:
        #     batch_loss = 0
        #     for idx in index:
        #         c = self.context_list[idx]
        #         r = self.super_reward[idx]
        #         optimizer.zero_grad()
        #         delta = self.func_super(c.cuda()) - r
        #         # delta = self.func_super(c) - r
        #         loss = delta * delta
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss += loss.item()
        #         tot_loss += loss.item()
        #         cnt += 1
        #         if cnt >= 1:
        #             return tot_loss / 1
        #     if batch_loss / length <= 1e-3:
        #         return batch_loss / length

        self.len += 1
        optimizer = optim.SGD(self.func_super.parameters(), lr=0.02, weight_decay=self.lamdba / self.len)
        if self.rewards_list is None:
            self.rewards_list = torch.from_numpy(rewards.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.super_reward = torch.tensor([super_reward], device='cuda', dtype=torch.float32)
        else:
            self.rewards_list = torch.cat((self.rewards_list, torch.from_numpy(rewards.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.super_reward = torch.cat((self.super_reward, torch.tensor([super_reward], device='cuda', dtype=torch.float32)))
        #if self.len % self.delay != 0:
        #   return 0
        for _ in range(40):
            self.func_super.zero_grad()
            optimizer.zero_grad()
            pred = self.func_super(self.rewards_list).view(-1)
            loss = self.loss_func_super(pred, self.super_reward)
            loss.backward()
            optimizer.step()
        return 0


import numpy as np
import argparse
import pickle
import os
import time
import torch
def max_sum_of_subarray(arr, n, k):
    max_sum = 0;
    for i in range(0, n-k+1):
        temp = 0;
        for j in range(i, i+k):
            temp += arr[j];

        if (temp > max_sum):
            max_sum = temp;

    return max_sum;

#if __name__ == '__main__':
torch.set_num_threads(9)
torch.set_num_interop_threads(9)
parser = argparse.ArgumentParser(description='NeuralUCB')



parser.add_argument('--size', default=1000, type=int, help='bandit size')
parser.add_argument('--super_arm_size', default=5, type=int, help='super arm size')
parser.add_argument('--dataset', default='movielens', metavar='DATASET')
parser.add_argument('--shuffle', type=bool, default=0, metavar='1 / 0', help='shuffle the data set or not')
parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularzation')
parser.add_argument('--hidden', type=int, default=100, help='network hidden size')
parser.add_argument('--style', default='ucb', metavar='ts|ucb', help='TS or UCB')



# args = parser.parse_args()
args, unknown = parser.parse_known_args()
use_seed = None if args.seed == 0 else args.seed
b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
bandit_info = '{}'.format(args.dataset)
K = args.super_arm_size
#budget = 5
l = NeuralUCBDiag(args.style, K, b.dim,np.shape(tf_idf_filtered_array)[0] , args.lamdba, args.nu, args.hidden)
lm = NeuralUCBDiagMono(K,args.lamdba, args.nu, 30)
ucb_info = '_{:.3e}_{:.3e}_{}'.format(args.lamdba, args.nu, args.hidden)
num_rounds = 1000
problem_model = MovielensProblemModel(num_rounds = num_rounds, budget = K, use_saved = False)


regrets = []
random_regrets = []
random_summ = 0
summ = 0
rew = 0
rewards_list = []
random_rewards_list = []
total_reward = 0
random_total_reward = 0
for t in range(min(args.size, b.size)):

    context = b.step(t+1)
    available_arms, exp_outcome_list, exp_outcome_list_all = problem_model.get_available_arms(context)
    
    # Assuming the modifications for l.func to accept batch inputs are in place
    indices = []
    ucb_cluster = []
    avg_ucb_cluster = []
    g_list_clusters = []
    
    for n, arms_of_cluster in enumerate(available_arms):
        # Extract all contexts for a given cluster and convert them to a suitable batch format
        cluster_contexts = np.array([arm.context for arm in arms_of_cluster])
        # Convert contexts to a PyTorch tensor in one go
        cluster_contexts_torch = torch.from_numpy(cluster_contexts.reshape(-1, 171)).to(device='cuda', dtype=torch.float32)
        # Predict rewards for all contexts in the cluster in one batch
        cluster_base_rewards = l.predict(cluster_contexts_torch).detach().cpu().numpy().flatten()
    
        # Find indices of the top K rewards
        top_k_indices = np.argpartition(cluster_base_rewards, -K)[-K:]
        indices.append(top_k_indices)
    
        # Select the top K rewards and their corresponding contexts
        top_k_rewards = cluster_base_rewards[top_k_indices]
        top_k_contexts = cluster_contexts[top_k_indices]
        top_k_contexts_tensor = torch.from_numpy(top_k_contexts).to(device='cuda', dtype=torch.float32)
        top_k_contexts_tensor = top_k_contexts_tensor.detach()
        # Further processing if needed
        predicted_rew_super = lm.func_super(torch.tensor(top_k_rewards, device='cuda', dtype=torch.float32))
        ucbs, g_list_cluster = l.get_ucb_cluster(top_k_contexts_tensor)
        g_list_clusters.append(g_list_cluster)
    
        # Calculate UCB and average for the cluster
        ucb_for_cluster = torch.add(ucbs, predicted_rew_super)
        ucb_cluster.append(ucb_for_cluster)
        avg_ucb_cluster.append(torch.mean(ucb_for_cluster).item())
        best_cluster_index = avg_ucb_cluster.index(max(avg_ucb_cluster))
        best_cluster_g_list = g_list_clusters[best_cluster_index]
        updating = l.update_params(best_cluster_g_list)
        selected_cluster_arms = available_arms[best_cluster_index][indices[best_cluster_index]]

    #print(sum(exp_outcome_list)/len(exp_outcome_list))
    #random_arm_select, arm_select = l.select(context,available_arms)
    #print(arm_select)
    rwd = problem_model.play_arms(selected_cluster_arms)
    rewards = rwd
    #random_rewards = problem_model.play_arms(random_arm_select)
    #random_super_reward = problem_model.get_total_reward(random_rewards)
    #if np.sum(rewards) >= 1:
    #  super_reward = 1
    #else:
    #  super_reward = 0
    super_reward = problem_model.get_total_reward(rewards)
    #print('Super reward is ' + str(super_reward))
    #random_total_reward += random_super_reward
    total_reward += super_reward
    rewards_list.append(total_reward)
    #random_rewards_list.append(random_total_reward)

    #if max_sum_of_subarray(rwd,len(rwd),K) >= 1:
    #  max_rew = 1
    #else:
    #  max_rew = 0

    #reg = max_rew - super_reward

    #reg = problem_model.get_regret(exp_outcome_list,K,arm_select,available_arms)
    rew_list = []
    for a in range(len(rewards)):
      rew_list.append(rewards[a].quality)
    rew_list = np.array(rew_list)
    #predicted_super_rew = lm.func_super(torch.from_numpy(rew_list.reshape(1, -1)).to(device='cuda', dtype=torch.float32))
    #concatenated_exp_outcome_list_all = []
    #for n in range(len(exp_outcome_list_all)):
    #    concatenated_exp_outcome_list += list(exp_outcome_list[n])
    best_super_reward = problem_model.get_optimal_super_arm_reward(exp_outcome_list_all, K)
    #print('Best super reward is ' + str(best_super_reward))
    reg = best_super_reward-super_reward
    if reg < 0:
        reg = 0




    summ+=reg
    #random_reg = problem_model.get_regret(exp_outcome_list,K,random_arm_select,available_arms)
    #random_summ +=random_reg
    #random_regrets.append(random_summ)

    loss = 0
    loss_super = 0
    regrets.append(summ)
    rew += super_reward
    # rew_list = []
    # for a in range(len(rewards)):
    #   rew_list.append(rewards[a].quality)
    if t<=num_rounds:
      if t%10 == 0:
        for i in range(len(rewards)):
          loss += l.train(selected_cluster_arms[i].context, rewards[i].quality)
        loss_super += lm.train(np.array(rew_list),super_reward)
        print('{}: {:.10f}, {:.3e},{:.3e}'.format(t, summ, loss_super,rew))
      else:
        for i in range(len(rewards)):
          loss += l.train(selected_cluster_arms[i].context, rewards[i].quality)
        loss_super += lm.train(np.array(rew_list),super_reward)
    #else:
    #    if t%100 == 0:
    #      for i in range(len(rewards)):
    #        loss += l.train(context[arm_select[i],:], rewards[i])
    #      loss_super += lm.train(np.array(rewards),super_reward)

    #if t % 10 == 0:
        #print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss_super, sig, ave_rwd))





print("--- %s seconds ---" % (time.time() - start_time))


path = '{}_{}_{}'.format(bandit_info, ucb_info, time.time())
fr = open(path,'w')
for i in regrets:
    fr.write(str(i))
    fr.write("\n")
fr.close()

