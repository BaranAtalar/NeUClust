"""Run one bandit algorithm on one dataset and save its cumulative regret and reward.

    python run.py --algo neuclust --dataset movielens
    python run.py --algo cnucb --dataset yelp --rounds 500 --seed 3
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from bandits import CCMAB, CNUCB, LinUCB, NeuralMAB, NeUClust
from environments import expected_reward, load_environment, super_arm_reward

ALGORITHMS = {"neuclust": NeUClust, "cnucb": CNUCB, "linucb": LinUCB, "neuralmab": NeuralMAB, "ccmab": CCMAB}

# Hyperparameters used for the experiments in the paper.
SETTINGS = {
    ("neuclust", "movielens"): dict(hidden=20, super_hidden=20, lr=1e-3, super_lr=1e-3, epochs=40,
                                    beta=0.01, clamp_super=True),
    ("neuclust", "yelp"): dict(hidden=100, super_hidden=30, lr=1e-3, super_lr=0.02, epochs=40,
                               beta=0.01, clamp_super=False),
    ("cnucb", "movielens"): dict(hidden=20, lr=1e-2, epochs=10, beta=1.0),
    ("cnucb", "yelp"): dict(hidden=100, lr=1e-3, epochs=40, beta=0.1),
    ("linucb", "movielens"): dict(alpha=1.0),
    ("linucb", "yelp"): dict(alpha=1.0),
    ("neuralmab", "movielens"): dict(lr=0.01, epochs=50),
    ("neuralmab", "yelp"): dict(lr=1e-3, epochs=30),
    ("ccmab", "movielens"): dict(),
    ("ccmab", "yelp"): dict(),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=ALGORITHMS, default="neuclust")
    parser.add_argument("--dataset", choices=["movielens", "yelp"], default="movielens")
    parser.add_argument("--rounds", type=int, default=1000, help="time horizon T")
    parser.add_argument("--k", type=int, default=5, help="number of base arms in a super arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, help="hidden layer size of the base arm network "
                        "(neuclust/cnucb; default: paper setting)")
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0, help="regularization (neuclust/cnucb)")
    parser.add_argument("--nu", type=float, default=1.0, help="exploration variance scale (neuclust/cnucb)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", type=Path, help="CSV file for the results "
                        "(default: results/<algo>_<dataset>_seed<seed>.csv)")
    return parser.parse_args()


def make_algorithm(args, env, rng):
    params = dict(SETTINGS[args.algo, args.dataset])
    if args.algo in ("neuclust", "cnucb"):
        params.update(lam=args.lam, nu=args.nu)
        if args.hidden is not None:
            params["hidden"] = args.hidden
    if args.algo == "ccmab":
        params.update(rounds=args.rounds, rng=rng)
    return ALGORITHMS[args.algo](env, args.k, args.device, **params)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = load_environment(args.dataset)
    if args.rounds > env.max_rounds:
        raise ValueError(f"--rounds can be at most {env.max_rounds} for {args.dataset}")
    algo = make_algorithm(args, env, rng)
    out = args.out or Path("results") / f"{args.algo}_{args.dataset}_seed{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{args.algo} on {args.dataset}: {env.num_arms} base arms, K={args.k}, "
          f"T={args.rounds}, device={args.device}")
    start = time.time()
    regret = reward = 0
    history = []
    for t in range(args.rounds):
        contexts = env.contexts(t)
        means = expected_reward(contexts)

        arms = algo.select(contexts)
        base_rewards = rng.binomial(1, means[arms]).astype(np.float64)
        super_reward = super_arm_reward(base_rewards)
        algo.update(contexts, arms, base_rewards, super_reward)

        best_means = np.partition(means, -args.k)[-args.k:]
        optimal_reward = super_arm_reward(rng.binomial(1, best_means))
        regret += max(optimal_reward - super_reward, 0)
        reward += super_reward
        history.append((t + 1, regret, reward))
        if (t + 1) % args.log_every == 0 or t + 1 == args.rounds:
            print(f"round {t + 1:5d}  cumulative regret {regret:5d}  cumulative reward {reward:5d}  "
                  f"({time.time() - start:.0f}s)")

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "cumulative_regret", "cumulative_reward"])
        writer.writerows(history)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
