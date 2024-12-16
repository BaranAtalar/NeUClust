# NeUClust
This repository contains our Pytorch implementation of NeUClust in the paper Neural Combinatorial Clustered Bandits for Recommendation Systems (accepted by AAAI 2025).

# Prerequisites
Pytorch and Cuda
*torch == 2.4.1
*numpy == 1.22.2
*pandas == 1.4.0
*scipy == 1.9.3
*sklearn == 1.3.2

# Usage
Use algorithm_dataset.py where algorithm = {neuclust, cnucb, linucb, NeuralMAB} and dataset = {movielens, yelp} to run experiments for that algorithm and dataset pair. movie_id_genre_dict_all_new.pkl, user_id_prefs_dict_all_new.pkl, user_ids_all_new.pkl, filtered_movie_ids_all_new.pkl are the files necessary to run MovieLens related experiments and for Yelp download dataset from https://www.yelp.com/dataset and X_prefs_clusters_yelp_20.pkl is needed to run experiments. 

# Command Line Arguments
*
