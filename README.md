# NeUClust
This repository contains our Pytorch implementation of NeUClust in the paper Neural Combinatorial Clustered Bandits for Recommendation Systems (accepted by AAAI 2025). To implement CN-UCB we make use of [NeuralUCB](https://github.com/uclaml/NeuralUCB), and for CC-MAB we use [ACC-UCB](https://github.com/Bilkent-CYBORG/ACC-UCB).

# Prerequisites
* Pytorch and Cuda
* torch == 2.4.1
* numpy == 1.22.2
* pandas == 1.4.0
* scipy == 1.9.3
* sklearn == 1.3.2

# Usage
Use algorithm_dataset.py where algorithm = {neuclust, cnucb, linucb, NeuralMAB, CC_MAB} and dataset = {movielens, yelp} to run experiments for that algorithm and dataset pair. movie_id_genre_dict_all_new.pkl, user_id_prefs_dict_all_new.pkl, user_ids_all_new.pkl, filtered_movie_ids_all_new.pkl are the files necessary to run MovieLens related experiments and for Yelp download files business_json, review_json, user_json from https://www.yelp.com/dataset and X_prefs_clusters_yelp_20.pkl is needed to run experiments. 

# Command Line Arguments
* --size: bandit algorithm time horizon
* --super_arm_size: number of arms in super arm
* --dataset: datasets
* --shuffle: to shuffle the dataset or not
* --seed: random seed for shuffle
* --nu: nu for control variance
* --lambda: lambda for regularization
* --hidden: base arm network hidden size

# Example Run
python3 neuclust_movielens.py --super_arm_size 5 --lambda 1 --nu 1 -size 1000
