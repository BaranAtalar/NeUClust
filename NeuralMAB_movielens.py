import numpy as np
import pickle
import pandas as pd
#import ProblemModel
#import random_algo
import model_context_reward
import torch
from torch.autograd import Variable
from scipy.stats import norm
from random import sample
import time
start_time = time.time()
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
# def find_node_containing_context(context, leaves):
#     for leaf in leaves:
#         if leaf.contains_context(context):
#             return leaf


class Arm:
    def __init__(self, unique_id, context,true_mean):
        self.true_mean = true_mean  # Only used by the benchmark
        self.unique_id = unique_id
        self.context = context


class Reward:
    def __init__(self,arm, quality):
        self.quality = quality
        self.context = arm.context
        self.arm = arm
from abc import ABC
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

    def get_optimal_super_rew(self, exp_out_list,budget):
        #df = self.df.loc[t]
        ld = sorted(exp_out_list,reverse=True)
        highest_means = ld[:budget]
        # algo_mean_prod = 0
        reward_sum = 0
        reward_list = [1.0 * np.random.binomial(1, mean) for mean in highest_means]
        # for arm in slate:
        #     algo_mean_prod += available_arms[arm.unique_id].true_mean
        # for mean in highest_means:
        #     bench_mean_prod += mean

        # return bench_mean_prod - algo_mean_prod
        for reward in reward_list:
            reward_sum += reward
        if reward_sum >= len(reward_list)*8/10:
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
        movies_df = pd.read_csv("movies.csv")
        ratings_df = pd.read_csv("ratings.csv")

        # remove ratings older than Jan 2015
        ratings_df = ratings_df[ratings_df.timestamp > 1563272000]

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

        movies_list = []
        genres_list = np.array([self.num_rounds,1])
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










def to_arr(t):
    return t.cpu().detach().numpy()

def to_tensor(arr):
    return Variable(torch.from_numpy(arr))

def context_to_mean_fun(context):
    count = np.count_nonzero(context)
    if count == 0:
        count = 1
    return 2 / (1 + np.exp(-np.sum(context)/(2*count))) - 1


"""
This class represents the CC-MAB algorithm.
"""
def step(t):
    #assert self.cursor < self.size
    #X_movie = np.zeros((20, 1))
    #selected_movie =random.choice(self.movie_select_list)
    #array_movie_select_list = np.array(self.movie_select_list)
    #indexx = np.where(array_movie_select_list == selected_movie)
    #self.movie_select_list.pop(indexx)
    movie_id_t = t
    #X_movie = self.movie_select_list[:,movie_id_t]
    X_movie = movie_id_genre_dict[:,movie_id_t]
    X_user = user_id_prefs_dict
    X = np.zeros((20,len(user_ids)))
    for j in range(len(user_ids)):
        X[:,j] = X_user[user_ids[j]]*X_movie

    return X

class NeuralC2UCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, budget, context_dim):  # Assumes a 1 x 1 x ... x 1 context space
        #global problem_model
        global movie_id_genre_dict
        global user_id_prefs_dict
        global user_ids
        global filtered_movie_ids
        self.context_dim = context_dim
        self.num_rounds = problem_model.num_rounds
        # self.hT = np.ceil(self.num_rounds ** (1 / (3 + context_dim)))
        # self.cube_length = 1 / self.hT
        self.budget = budget
        self.problem_model = problem_model
        #self.real = real

    # def get_hypercube_of_context(self, context):
    #     return tuple((context / self.cube_length).astype(int))

    
    def run_algorithm(self):
        context_list = []
        super_reward_list = []
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        
        

        #if not self.real:
        #    return total_reward_arr, regret_arr

        # initialize the neural network
        nn = model_context_reward.linearRegression(self.context_dim, 1).cuda()

        num_arms = len(user_id_prefs_dict)
        # reward_func = model.linearRegression(num_arms * 2, 1)
        # x_train = []
        # y_train = []
        total_reward = 0
        total_regret = 0
        reward_list = []
        regret_list = []
        for t in range(1, self.num_rounds+1):
            #print("------ neural", t, "------")
            # get contexts
            context = step(t)
            available_arms,exp_out_list = self.problem_model.get_available_arms(context)
            contexts_t = []
            for arm in available_arms:
                contexts_t.append(arm.context)
            contexts_t = np.stack(contexts_t)
            
            inputs = to_tensor(contexts_t).cuda()
            outputs = nn(inputs.float())
            r_t_hat = to_arr(outputs)
            # print("oracle:", r_t_hat.reshape(-1))
            r_t_hat = np.asarray(r_t_hat).reshape(-1).tolist()
            expected_r_t = np.prod(contexts_t, axis=1)
            
            # for row in contexts_t:
            #     print(context_to_mean_fun(row))
            # print("predicted rewards:", r_t_hat)
            # print("expected rewards:", expected_r_t.tolist())

            # print("[Neural-C2UCB] learned quality:\n", r_t_hat)
            slate = []
            arm_indices = self.problem_model.oracle(self.budget, r_t_hat)
            # print("oracle selection:", arm_indices)
            # print("[Neural-C2UCB] arm_indices:\n", arm_indices)
            for index in arm_indices:
                selected_arm = available_arms[index]
                slate.append(selected_arm)


            # slate = random_algo.sample(available_arms, self.budget)

            rewards = self.problem_model.play_arms(slate)  # Returns a list of Reward objects

            qualities = []
            for reward in rewards:
                qualities.append(reward.quality)
            #print(qualities)
            # print("oracle:", arm_indices, qualities)

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards)
            # print("[Neural-C2UCB] regret:", self.problem_model.get_regret(t, self.budget, slate))
            h = self.problem_model.get_optimal_super_rew(exp_out_list, self.budget) - self.problem_model.get_total_reward(rewards)
            if h < 0:
                h = 0
            regret_arr[t - 1] = h
            # print("reward:", total_reward_arr[t - 1], "regret:", regret_arr[t - 1])

            # train the network
            xs = []
            ys = []
            for reward in rewards:
                x = reward.context
                y = reward.quality
                xs.append(x)
                ys.append(y)

            xs = np.array(xs)
            ys = np.array(ys).reshape(-1, 1)

            # print(xs.shape)
            # print(ys.shape)

            if t == 1:
                x_train = xs
                y_train = ys
            else:
                x_train = np.vstack((x_train, xs))
                x_train = np.array(x_train, dtype=np.float32)
                y_train = np.vstack((y_train, ys))
                y_train = np.array(y_train, dtype=np.float32)

            #print(x_train.shape, y_train.shape)
            
            # for i in range(self.budget):
            #     context_list.append(torch.from_numpy(x_train[i,:].reshape(1, -1)).float())
            #     super_reward_list.append(y_train[i,:])
            #     model_context_reward_new.train(nn, self.context_dim, 1, 0.01, 3, x_train[i,:], y_train[i,:],context_list,super_reward_list)       
            
            model_context_reward.train(nn, self.context_dim, 1, 0.01, 50, x_train, y_train) 

            # print(to_arr(nn.linear.weight).reshape(-1, 1))
            # print(contexts_t)
            total_reward += self.problem_model.get_total_reward(rewards)
            reward_list.append(total_reward)
            total_regret += h
            regret_list.append(total_regret)
            if t<=1000:
              if t%10 == 0:
                print('{}: {:.10f}, {:.3e}'.format(t, total_regret, total_reward))

        return total_reward_arr, regret_arr, reward_list, regret_list

neuralc2ucb_algo = NeuralC2UCB(problem_model, budget = 5, context_dim = 20)
total_rew,total_reg,reward_list,regret_list = neuralc2ucb_algo.run_algorithm()
print("--- %s seconds ---" % (time.time() - start_time))

    
