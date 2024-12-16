# -*- coding: utf-8 -*-
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
        super().__init__(num_rounds)
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

        arm_list = []
        exp_out_list = []
        length = len(user_ids)
        for i in range(length):
            context_arm = context[:,i]
            exp_outcome = context_to_mean_fun(context_arm)
            arm_list.append(Arm(len(arm_list), np.array(context[:,i]), exp_outcome))
            exp_out_list.append(exp_outcome)
        return arm_list,exp_out_list

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
        reward_list = [1.0 * np.random.binomial(1, mean) for mean in highest_means]
        reward_sum = 0
        for reward in reward_list:
            reward_sum += reward
        #return reward_sum/len(reward_list)
        if reward_sum >= len(rewards)*8/10:
            return 1
        return 0
        
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
        #return reward_sum/len(rewards)

    def play_arms(self, slate):
        reward_list = [Reward(arm, 1.0 * np.random.binomial(1, arm.true_mean)) for arm in slate]
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

#movie_id_genre_dict,user_id_prefs_dict,user_ids,filtered_movie_ids = problem_model.initialize_dataset()

with open('movie_id_genre_dict_all_new.pkl', 'rb') as file:
      
    # Call load method to deserialze
    movie_id_genre_dict = pickle.load(file)

with open('user_id_prefs_dict_all_new.pkl', 'rb') as file:
      
    # Call load method to deserialze
    user_id_prefs_dict = pickle.load(file)

with open('user_ids_all_new.pkl', 'rb') as file:
      
    # Call load method to deserialze
    user_ids = pickle.load(file)

with open('filtered_movie_ids_all_new.pkl', 'rb') as file:
      
    # Call load method to deserialze
    filtered_movie_ids = pickle.load(file)

# rng = np.random.default_rng()
# rng.shuffle(movie_id_genre_dict,axis = 1)

# with open('movie_id_genre_dict_shuff_all.pkl', 'wb') as file:
      
#     # A new file will be created
#     pickle.dump(movie_id_genre_dict, file)


# if __name__ == '__main__':
#     test = MovielensProblemModel(250, 50, 100, False, 10, saved_file_name="temp_movielens.hdf5")

#     with open('problem_model_test_pickle', 'wb') as output:
#         pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)

#     # df['']
#     print('donerooni')

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np



class Bandit_multi:
    def __init__(self, name, is_shuffle=False, seed=None):
        # Fetch data
        global df
        global movie_id_genre_dict
        global user_id_prefs_dict
        global filtered_movie_ids 
        global user_ids
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
        elif name == 'gowalla':
            X = df
        elif name == 'movielens':
            X_movie = movie_id_genre_dict
            self.X_user = user_id_prefs_dict
        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        if is_shuffle:
            self.X= shuffle(X, random_state=seed)
            df = self.X
        else:
            #self.X, self.y = X, y
            #self.X = X
            if name == "movielens":
                self.X_movie = X_movie
                self.X_user = user_id_prefs_dict
        # generate one_hot coding:
        #self.y_arm = OrdinalEncoder(
        #    dtype=np.int).fit_transform(self.y.values.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = 1000
        #self.n_arm = np.max(self.y_arm) + 1
        #self.dim = self.X.shape[1] * self.n_arm
        self.dim = 20
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
        movie_id_t = t
        #X_movie = self.movie_select_list[:,movie_id_t]
        X_movie = movie_id_genre_dict[:,movie_id_t]
        X_user = self.X_user
        X = np.zeros((20,len(user_ids)))
        for j in range(len(user_ids)):
            X[:,j] = X_user[user_ids[j]]*X_movie

        return X











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


# class NNBaseClass(LightningModule):
#     def __init__(self, lr: float = 0.001, optimizer='adam', n_class=-1):
#         super().__init__()
#         self._lr = lr
#         self._optimizer = optimizer
#         self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)

#     def configure_optimizers(self):
#         if self._optimizer == 'adam':
#             optimizer = Adam(params=self.parameters(), lr=self._lr, betas=(0.9, 0.95))
#         elif self._optimizer == 'sgd':
#             optimizer = SGD(params=self.parameters(), lr=self._lr)
#         else:
#             raise RuntimeError
#         return optimizer

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
    def __init__(self, dim = 20, hidden_size=10):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):

        return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiag:
    def __init__(self, style, K, dim, num_users, lamdba=1, nu=1, hidden=10):
        self.func = extend(Network(dim, hidden_size=hidden).cuda())
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
        self.loss_func = nn.MSELoss()
        self.len = 0

    def select(self, context, available_arms):
        
        tensor = torch.from_numpy(np.transpose(context)).float().cuda()
        # tensor = torch.from_numpy(np.transpose(context)).float()
        #for arm in available_arms:
        #    print(arm.unique_id)
        
        mu = self.func(tensor)
        sum_mu = torch.sum(mu)
        
        with backpack(BatchGrad()):
            sum_mu.backward()
        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
        sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), 0.1*sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + 1*sigma.view(-1)
        #print(sample_r)
        #arm = torch.argmax(sample_r)
        values, super_arm = torch.topk(sample_r,self.K)
        for arm in super_arm:
          self.U += g_list[arm] * g_list[arm]
        super_arm_index = []
        for arm in super_arm:
           super_arm_index.append(available_arms[arm])
        arr = np.arange(self.num_users)
        chosen_arms = random.choices(list(arr), k = self.K)
        for arm in chosen_arms:
            self.U_random += g_list[arm] * g_list[arm]
        super_arm_index_random = []
        for arm in chosen_arms:
            super_arm_index_random.append(available_arms[arm])
        return super_arm_index_random, super_arm_index
        
        # g_list = []
        # sampled = []
        # ave_sigma = 0
        # ave_rew = 0 
        # for fx in mu:
        #     #print(fx)
        #     self.func.zero_grad()
        #     fx.backward(retain_graph=True)
        #     g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
        #     g_list.append(g)
        #     sigma2 = self.lamdba * self.nu * g * g / self.U
        #     sigma = torch.sqrt(torch.sum(sigma2))

        #     sample_r = fx.item() + 0.01*sigma.item()

        #     sampled.append(sample_r)
        #     ave_sigma += sigma.item()
        #     ave_rew += sample_r
        # sampled = np.array(sampled)
        # super_arm = self.problem_model.oracle(self.K,sampled)
        # for arm in super_arm:
        #   self.U += g_list[arm] * g_list[arm]
        # super_arm_index = []
        # for arm in super_arm:
        #   super_arm_index.append(available_arms[arm])
        # arr = np.arange(self.num_users)
        # chosen_arms = random.choices(list(arr), k = 5)
        # for arm in chosen_arms:
        #    self.U_random += g_list[arm] * g_list[arm]
        # super_arm_index_random = []
        # for arm in chosen_arms:
        #    super_arm_index_random.append(available_arms[arm])
        # return super_arm_index_random, super_arm_index, ave_sigma, ave_rew    


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
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba / self.len)
        if self.context_list is None:
            self.context_list = torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.reward = torch.tensor([reward], device='cuda', dtype=torch.float32)
        else:
            self.context_list = torch.cat((self.context_list, torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.reward = torch.cat((self.reward, torch.tensor([reward], device='cuda', dtype=torch.float32)))
        #if self.len % self.delay != 0:
        #    return 0
        for _ in range(10):
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
    def __init__(self, dim = 5, hidden_size=5):
        super(NetworkMono, self).__init__()
        # self.pre_monotonic = InterpretableLayer(dim)
        # self.pre_monotonic.weight.data.normal_(0.0, 1 / sqrt(nb_neuron_inter_layer))

        # self.monotonic = nn.Sequential(
        #       MonotonicLayer(dim, hidden_size, fn='tanh_p1'),
        #       nn.ReLU(),
        #       MonotonicLayer(hidden_size, 1, fn='tanh_p1'),
        # )

        # self.output = InterpretableLayer(1)
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        #x = self.high_level_feats(x)
        # x = self.pre_monotonic(x)
        # x = self.monotonic(x)
        # x = self.output(x)
        # return x
        return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiagMono:
    def __init__(self, dim=5, lamdba=1, nu=1, hidden=5):
        self.func_super = extend(NetworkMono(dim, hidden_size=hidden).cuda())
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
        self.loss_func_super = nn.MSELoss()


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
        optimizer = optim.SGD(self.func_super.parameters(), lr=1e-1, weight_decay=self.lamdba / self.len)
        if self.rewards_list is None:
            self.rewards_list = torch.from_numpy(rewards.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.super_reward = torch.tensor([super_reward], device='cuda', dtype=torch.float32)
        else:
            self.rewards_list = torch.cat((self.rewards_list, torch.from_numpy(rewards.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.super_reward = torch.cat((self.super_reward, torch.tensor([super_reward], device='cuda', dtype=torch.float32)))
        #if self.len % self.delay != 0:
        #   return 0
        for _ in range(10):
            self.func_super.zero_grad()
            optimizer.zero_grad()
            pred = self.func_super(self.rewards_list).view(-1)
            loss = self.loss_func_super(pred, self.super_reward)
            loss.backward()
            optimizer.step()
        return 0

#from data_multi import Bandit_multi
#from learner_diag import NeuralUCBDiag
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
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
parser = argparse.ArgumentParser(description='NeuralUCB')



parser.add_argument('--size', default=1000, type=int, help='bandit size')
parser.add_argument('--super_arm_size', default=5, type=int, help='super arm size')
parser.add_argument('--dataset', default='movielens', metavar='DATASET')
parser.add_argument('--shuffle', type=bool, default=0, metavar='1 / 0', help='shuffle the data set or not')
parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularzation')
parser.add_argument('--hidden', type=int, default=20, help='network hidden size')
parser.add_argument('--style', default='ucb', metavar='ts|ucb', help='TS or UCB')



# args = parser.parse_args()
args, unknown = parser.parse_known_args()
use_seed = None if args.seed == 0 else args.seed
b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
bandit_info = '{}'.format(args.dataset)
K = args.super_arm_size
#budget = 5
l = NeuralUCBDiag(args.style, K, b.dim,len(user_id_prefs_dict) , args.lamdba, args.nu, 20)
#lm = NeuralUCBDiagMono(K,args.lamdba, args.nu, args.hidden)
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
    available_arms,exp_outcome_list = problem_model.get_available_arms(context)
    #print(sum(exp_outcome_list)/len(exp_outcome_list))
    random_arm_select, arm_select = l.select(context,available_arms)
    #print(arm_select)
    rwd = problem_model.play_arms(arm_select)
    rewards = rwd
    random_rewards = problem_model.play_arms(random_arm_select)
    random_super_reward = problem_model.get_total_reward(random_rewards)
    #if np.sum(rewards) >= 1:
    #  super_reward = 1
    #else:
    #  super_reward = 0
    super_reward = problem_model.get_total_reward(rewards)
    random_total_reward += random_super_reward
    total_reward += super_reward
    rewards_list.append(total_reward)
    random_rewards_list.append(random_total_reward)
    
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
    best_super_reward = problem_model.get_optimal_super_arm_reward(exp_outcome_list, K)
    reg = (best_super_reward-super_reward)
    if reg < 0:
        reg = 0
    

    
    
    
    summ+=reg
    random_reg = problem_model.get_regret(exp_outcome_list,K,random_arm_select,available_arms)
    random_summ +=random_reg
    random_regrets.append(random_summ)
    
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
          loss += l.train(context[:,arm_select[i].unique_id], rewards[i].quality)
        #loss_super += lm.train(np.array(rew_list),super_reward)
        print('{}: {:.10f}, {:.3e},{:.3e}'.format(t, summ, loss_super,rew))
      else:
        for i in range(len(rewards)):
          loss += l.train(context[:,arm_select[i].unique_id], rewards[i].quality)
        #loss_super += lm.train(np.array(rew_list),super_reward)            
    #else:
    #    if t%100 == 0:
    #      for i in range(len(rewards)):  
    #        loss += l.train(context[arm_select[i],:], rewards[i])
    #      loss_super += lm.train(np.array(rewards),super_reward)
    
    #if t % 10 == 0:
        #print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss_super, sig, ave_rwd))



import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(np.linspace(1,num_rounds,num_rounds),np.array(rewards_list))
# plt.plot(np.linspace(1,num_rounds,num_rounds),np.array(random_rewards_list))
# plt.legend(["NeuralUCBMono","Random"])

# plt.title("Super arm reward vs iterations")
# plt.xlabel("Iterations")
# plt.ylabel("Reward")

# plt.figure(2)
# plt.plot(np.linspace(1,num_rounds,num_rounds),np.array(regrets))
# plt.plot(np.linspace(1,num_rounds,num_rounds),np.array(random_regrets))
# plt.legend(["NeuralUCBMono","Random"])

# plt.title("Super arm regret vs iterations")
# plt.xlabel("Iterations")
# plt.ylabel("Regret")


with open('rebuttal_reg_cnucb_movielens_1.pkl', 'wb') as file:

    # A new file will be created
    pickle.dump(regrets, file)

with open('rebuttal_rew_cnucb_movielens_1.pkl', 'wb') as file:

    # A new file will be created
    pickle.dump(rewards_list, file)

# with open('file_rew_rand_final_9.pkl', 'wb') as file:
      
#     # A new file will be created
#     pickle.dump(random_rewards_list, file)
    
# with open('file_reg_rand_final_9.pkl', 'wb') as file:
      
#     # A new file will be created
#     pickle.dump(random_regrets, file)


import matplotlib.pyplot as plt
num_rounds = 1000
plt.figure(1)


print("--- %s seconds ---" % (time.time() - start_time))

path = '{}_{}_{}'.format(bandit_info, ucb_info, time.time())
fr = open(path,'w')
for i in regrets:
    fr.write(str(i))
    fr.write("\n")
fr.close()