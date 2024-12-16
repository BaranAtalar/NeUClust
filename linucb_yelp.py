
import time
start_time = time.time()
import pickle
import pandas as pd
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

# business_data = pd.read_pickle("business_data.pkl")
# review_data = pd.read_pickle("review_data.pkl")
# user_data = pd.read_pickle("user_data.pkl")
# X_prefs_clusters = pd.read_pickle("X_prefs_clusters_yelp.pkl")

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

import numpy as np
def get_available_arms(context):
    #if self.h5_file is None:
    #    self.h5_file = h5py.File(self.saved_file_name, "r")

    #t = t - 1
    # Construct a list of Arm objects (i.e., available edges)
    #context_dataset = self.h5_file[f"{t}"]["context_dataset"]
    #exp_outcome_dataset = self.h5_file[f"{t}"]["mean_dataset"]

    arm_list = []
    exp_out_list = []
    length = np.shape(tf_idf_filtered_array)[0]
    for i in range(length):
        context_arm = context[:,i]
        exp_outcome = context_to_mean_fun(context_arm)
        arm_list.append(Arm(len(arm_list), np.array(context[:,i]), exp_outcome))
        exp_out_list.append(exp_outcome)
    return arm_list,exp_out_list


class Arm:
    def __init__(self, unique_id, context,true_mean):
        self.true_mean = true_mean  # Only used by the benchmark
        self.unique_id = unique_id
        self.context = context






# Create class object for a single linear ucb disjoint arm
class linucb_disjoint_arm():

    def __init__(self, arm_index, d, alpha):

        # Track arm index
        self.arm_index = arm_index

        # Keep track of alpha
        self.alpha = alpha

        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros(d).reshape([-1,1])

    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        #A_inv = np.linalg.inv(self.A)

        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        #self.theta = np.dot(A_inv, self.b)
        #self.theta, res, rank,s = np.linalg.lstsq(self.A,self.b,rcond=None)
        self.theta = np.linalg.solve(self.A,self.b)
        #print(np.shape(self.theta))
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        #print(np.shape(x))
        # Find ucb based on p formulation (mean + std_dev)
        # p is (1 x 1) dimension vector
        #print(np.shape(np.linalg.lstsq(self.A,x,rcond=None)[0]))
        p = float(np.dot(self.theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, (np.linalg.solve(self.A,x)))))

        return p

    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)

        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x
#Here is the Class object for the LinUCB policy for K number of arms. It has two main methods:
budget = 5
#Initiation: Create a list of K linucb_disjoint_arm objects
#Arm selection: Choose arm based on the arm with the highest UCB for a given time step.
class linucb_policy():

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = d, alpha = alpha) for i in range(K_arms)]

    def select_arm(self, x_array,available_arms):
        # Initiate ucb to be 0
        #highest_ucb = -1

        # Track index of arms to be selected on if they have the max UCB.
        arm_ucb_list = []

        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array[:,arm_index])
            #print(arm_ucb)
            arm_ucb_list.append(arm_ucb)
            # If current arm is highest than current highest_ucb
            #if arm_ucb > highest_ucb:


                # Set new max ucb
            #    highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
            #    candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            #if arm_ucb == highest_ucb:

            #    candidate_arms.append(arm_index)
        arm_ucb_arr = np.array(arm_ucb_list)
        arm_ucb_arr = np.reshape(arm_ucb_arr,np.shape(tf_idf_filtered_array)[0])
        #print(np.shape(arm_ucb_arr))
        arm_indices = np.argpartition(arm_ucb_arr,-budget)[-budget:]
        # Choose based on candidate_arms randomly (tie breaker)
        #chosen_arm = np.random.choice(candidate_arms)
        super_arm = []
        for arm in arm_indices:
            super_arm.append(available_arms[arm])




        return super_arm





linucb_policy_object = linucb_policy(K_arms = 52268, d = 171, alpha = 1)
num_rounds = 1000



def step(t):
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
    X_final = np.zeros((np.shape(tf_idf_filtered_array)[1],np.shape(tf_idf_filtered_array)[0]))
    cnt = 0
    for j in range(len(X_prefs_clusters)):
        for i in range(len(X_prefs_clusters[j])):
            X_final[:,cnt] = X_prefs_clusters[j][i]*user_prefs_list
            cnt = cnt+1
    return X_final

def context_to_mean_fun(context):
    """
    """
    count = np.count_nonzero(context)
    if count == 0:
        count = 1
    return 2 / (1 + np.exp(-np.sum(context)/(2*count))) - 1





class Reward:
    def __init__(self,arm, quality):
        self.quality = quality
        self.context = arm.context
        self.arm = arm

def play_arms(slate):
    reward_list = [Reward(arm, 1.0 * np.random.binomial(1, arm.true_mean)) for arm in slate]
    return reward_list

def get_total_reward(rewards):


    reward_sum = 0
    for reward in rewards:
        reward_sum += reward.quality  # Total reward is lin sum
    if reward_sum >= len(rewards)*8/10:
        return 1
    return 0
    # return reward_sum/len(rewards)


def get_optimal_super_reward(exp_out_list,budget,available_arms):
    #df = self.df.loc[t]
    ld = sorted(exp_out_list,reverse=True)
    highest_means = ld[:budget]
    optimal_rewards_list = [np.random.binomial(1, mean) for mean in highest_means]
    #print(highest_means)
    #algo_mean_prod = 0
    bench_mean_prod = 0
    # for arm in slate:
    #     #print(available_arms[arm.unique_id].true_mean)
    #     algo_mean_prod += available_arms[arm.unique_id].true_mean
    for rew in optimal_rewards_list:
        bench_mean_prod += rew
    if bench_mean_prod >= len(optimal_rewards_list)*8/10:
        return 1
    return 0
    #return bench_mean_prod

total_reward = 0
rewards_list = []
summ = 0
rew = 0
regrets = []
oracle_total_reward = 0
oracle_rewards_list = []
for t in range(num_rounds):
    context = step(t+1)
    available_arms,exp_outcome_list = get_available_arms(context)
#    ld = sorted(exp_outcome_list,reverse=True)
#    highest_means = ld[:budget]
    oracle_arm_indices = np.argpartition(exp_outcome_list,-budget)[-budget:]
    oracle_super_arm = []
    for arm in oracle_arm_indices:
        oracle_super_arm.append(available_arms[arm])
    rewards_oracle = play_arms(oracle_super_arm)
    oracle_super_reward = get_total_reward(rewards_oracle)

    arm_select = linucb_policy_object.select_arm(context,available_arms)
    rewards = play_arms(arm_select)

    #for reward in rewards:
    #    print(reward.quality)
    j=0
    for arm in arm_select:
        linucb_policy_object.linucb_arms[arm.unique_id].reward_update(rewards[j].quality, context[:,arm.unique_id])
        j +=1
    oracle_total_reward += oracle_super_reward
    oracle_rewards_list.append(oracle_total_reward)
    super_reward = get_total_reward(rewards)
    total_reward += super_reward
    rewards_list.append(total_reward)
    #reg = get_regret(exp_outcome_list,budget,arm_select,available_arms)
    optimal_super_rew = get_optimal_super_reward(exp_outcome_list, budget, available_arms)
    #print(reg)
    h = optimal_super_rew-super_reward
    if h < 0:
        h = 0
    summ+=h
    regrets.append(summ)
    if t<=1000:
      if t%10 == 0:
        print('{}: {:.10f}, {:.3e}'.format(t, summ, total_reward))
print("--- %s seconds ---" % (time.time() - start_time))
