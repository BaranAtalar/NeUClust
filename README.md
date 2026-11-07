# NeUClust

PyTorch implementation of **NeUClust** from [Neural Combinatorial Clustered Bandits for Recommendation Systems](https://arxiv.org/abs/2410.14586) (AAAI 2025), together with the baselines used in the paper: CN-UCB, LinUCB, NeuralMAB and CC-MAB. CN-UCB builds on [NeuralUCB](https://github.com/uclaml/NeuralUCB) and CC-MAB on [ACC-UCB](https://github.com/Bilkent-CYBORG/ACC-UCB).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.0+ are required. The code runs on a GPU when one is available and falls back to the CPU otherwise.

## Quick start

The MovieLens data is included, so this runs right away:

```bash
python run.py --algo neuclust --dataset movielens
```

Progress is printed while it runs, and the cumulative regret and reward of every round are written to `results/neuclust_movielens_seed0.csv`.

| `--algo` | Method |
|---|---|
| `neuclust` | NeUClust (ours) |
| `cnucb` | CN-UCB |
| `linucb` | LinUCB (disjoint, one model per base arm) |
| `neuralmab` | NeuralMAB (greedy, learned reward model) |
| `ccmab` | CC-MAB |

## Datasets

**MovieLens** (`data/movielens.npz`, included). The base arms are 10,168 users who rated at least 200 movies since 2015. Each user is a 20-dimensional vector of their average rating per genre. In round *t* a movie arrives, and the context of each user is their preference vector multiplied element-wise by the movie's one-hot genre vector.

**Yelp** (needs a one-time preparation step). Download the [Yelp Open Dataset](https://www.yelp.com/dataset), extract it, and run

```bash
python prepare_yelp.py /path/to/yelp_dataset
```

This reads `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` and writes `data/yelp.npz` (about 4 MB) in a few minutes. The base arms are the 52,268 restaurants, described by the TF-IDF of their categories (171 terms) and grouped into 20 KMeans clusters. In round *t* a user arrives, and the context of each restaurant is its feature vector multiplied by the user's average rating per category. After preparation:

```bash
python run.py --algo neuclust --dataset yelp
```

For both datasets, base arm *i* is clicked with probability 2σ(Σx / (2·nnz(x))) − 1, where σ is the sigmoid and *x* the arm's context. A super arm earns reward 1 when at least 80% of its base arms are clicked. Regret compares this reward with the super arm made of the *K* arms with the highest click probabilities.

## Options

| Flag | Default | Description |
|---|---|---|
| `--algo` | `neuclust` | algorithm to run (see table above) |
| `--dataset` | `movielens` | `movielens` or `yelp` |
| `--rounds` | `1000` | time horizon *T* |
| `--k` | `5` | number of base arms in a super arm |
| `--seed` | `0` | random seed (rewards, network initialization, sampling) |
| `--hidden` | paper setting | hidden layer size of the base arm network (NeUClust, CN-UCB) |
| `--lambda` | `1.0` | regularization parameter λ (NeUClust, CN-UCB) |
| `--nu` | `1.0` | exploration variance scale ν (NeUClust, CN-UCB) |
| `--device` | `cuda` if available | PyTorch device |
| `--log-every` | `50` | print progress every this many rounds |
| `--out` | `results/<algo>_<dataset>_seed<seed>.csv` | output file |

All other hyperparameters (learning rates, training epochs, exploration coefficients) are the ones used in the paper. They are listed in `SETTINGS` at the top of `run.py`.

To run every method over several seeds:

```bash
for algo in neuclust cnucb linucb neuralmab ccmab; do
  for seed in 0 1 2 3 4; do
    python run.py --algo $algo --dataset movielens --seed $seed
  done
done
```

## Code layout

| File | Contents |
|---|---|
| `run.py` | command line interface, experiment loop and paper hyperparameters |
| `bandits.py` | NeUClust and the baselines |
| `environments.py` | reward model and the MovieLens and Yelp environments |
| `prepare_yelp.py` | builds `data/yelp.npz` from the Yelp Open Dataset |

## Notes

This version fixes two bugs in the original experiment scripts:

* In NeUClust, the exploration matrix was updated once for every cluster in every round. It is now updated once per round with the gradients of the super arm that was played.
* In CN-UCB on Yelp, ties in the UCB values could make it play more than *K* arms. It now always plays exactly *K*.

Because of these fixes, numbers can differ slightly from the ones reported in the paper.

## Citation

```bibtex
@inproceedings{atalar2025neuclust,
  title     = {Neural Combinatorial Clustered Bandits for Recommendation Systems},
  author    = {Atalar, Baran and Joe-Wong, Carlee},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {15},
  pages     = {15417--15426},
  year      = {2025},
  doi       = {10.1609/aaai.v39i15.33692}
}
```
