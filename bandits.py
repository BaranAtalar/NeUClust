"""NeUClust and the baselines compared against it in the paper.

Every algorithm implements
    select(contexts)                                    -> indices of the K base arms to play
    update(contexts, arms, base_rewards, super_reward)  -> learn from the observed feedback
where `contexts` is the (num_arms, dim) context matrix of the current round.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap


def mlp(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class MonotoneLinear(nn.Module):
    """Linear layer whose weights are passed through a sigmoid, so they are always positive."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.weight)
        nn.init.uniform_(self.bias, -1 / in_features, 1 / in_features)

    def forward(self, x):
        return x @ torch.sigmoid(self.weight).T + self.bias


def monotone_mlp(dim, hidden):
    """Network that maps the K base arm rewards to the super arm reward, monotone in each input."""
    return nn.Sequential(MonotoneLinear(dim, hidden), nn.ReLU(), MonotoneLinear(hidden, 1))


class Regressor:
    """Network trained with full-batch SGD on all samples seen so far, one sample at a time."""

    def __init__(self, net, lr, epochs, lam, device, clamp=False):
        self.net = net.to(device)
        self.lr, self.epochs, self.lam, self.clamp = lr, epochs, lam, clamp
        self.device = device
        self.inputs, self.targets = [], []

    def __call__(self, x):
        return self.net(torch.as_tensor(x, dtype=torch.float32, device=self.device)).view(-1)

    def add_sample_and_train(self, x, y):
        self.inputs.append(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        self.targets.append(float(y))
        inputs = torch.stack(self.inputs)
        targets = torch.tensor(self.targets, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.lam / len(self.targets))
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.net(inputs).view(-1), targets)
            loss.backward()
            optimizer.step()
            if self.clamp:
                with torch.no_grad():
                    for p in self.net.parameters():
                        p.clamp_(min=0)


class NeuralUCB(Regressor):
    """Neural network reward estimate plus the diagonal NeuralUCB exploration bonus."""

    def __init__(self, dim, hidden, lr, epochs, lam, nu, beta, device):
        super().__init__(mlp(dim, hidden), lr, epochs, lam, device)
        self.nu, self.beta = nu, beta
        num_params = sum(p.numel() for p in self.net.parameters())
        self.U = lam * torch.ones(num_params, device=device)

    def gradients(self, x):
        """Per-sample gradients of the network output w.r.t. all parameters, shape (n, num_params)."""
        params = {k: v.detach() for k, v in self.net.named_parameters()}

        def output(p, sample):
            return functional_call(self.net, p, (sample.unsqueeze(0),)).squeeze()

        grads = vmap(grad(output), in_dims=(None, 0))(params, x)
        return torch.cat([g.reshape(len(x), -1) for g in grads.values()], dim=1)

    def ucb(self, x, max_elements=2**26):
        """mean + beta * sigma for every row of x, in chunks so the gradients stay below 256 MB."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        ucbs = []
        for chunk in torch.split(x, max(1, max_elements // len(self.U))):
            with torch.no_grad():
                mu = self.net(chunk).view(-1)
            g = self.gradients(chunk)
            sigma = torch.sqrt(torch.sum(self.lam * self.nu * g * g / self.U, dim=1))
            ucbs.append(mu + self.beta * sigma)
        return torch.cat(ucbs)

    def update_confidence(self, x):
        """Add the gradients of the played arms to the diagonal design matrix U."""
        for g in self.gradients(torch.as_tensor(x, dtype=torch.float32, device=self.device)):
            self.U += g * g


class NeUClust:
    """Neural combinatorial clustered bandit (ours).

    In each cluster the K arms with the highest predicted reward form a candidate super arm.
    Its score is the mean UCB of those arms plus the super arm reward predicted by a monotone
    network from their predicted rewards; the best scoring cluster's super arm is played.
    """

    def __init__(self, env, k, device, hidden, super_hidden, lr, super_lr, epochs, beta, lam, nu,
                 clamp_super):
        self.k = k
        self.clusters = env.clusters
        self.base = NeuralUCB(env.dim, hidden, lr, epochs, lam, nu, beta, device)
        self.super = Regressor(monotone_mlp(k, super_hidden), super_lr, epochs, lam, device,
                               clamp=clamp_super)

    def select(self, contexts):
        scores, candidates = [], []
        for arms in self.clusters:
            with torch.no_grad():
                predicted = self.base(contexts[arms]).cpu().numpy()
            top = np.argpartition(predicted, -self.k)[-self.k:]
            ucb = self.base.ucb(contexts[arms[top]])
            with torch.no_grad():
                super_estimate = self.super(predicted[top])
            scores.append(torch.mean(ucb + super_estimate).item())
            candidates.append(arms[top])
        best = candidates[int(np.argmax(scores))]
        self.base.update_confidence(contexts[best])
        return best

    def update(self, contexts, arms, base_rewards, super_reward):
        for arm, reward in zip(arms, base_rewards):
            self.base.add_sample_and_train(contexts[arm], reward)
        self.super.add_sample_and_train(base_rewards, super_reward)


class CNUCB:
    """Combinatorial neural UCB (Hwang et al., 2023): play the K arms with the highest UCB."""

    def __init__(self, env, k, device, hidden, lr, epochs, beta, lam, nu):
        self.k = k
        self.model = NeuralUCB(env.dim, hidden, lr, epochs, lam, nu, beta, device)

    def select(self, contexts):
        arms = torch.topk(self.model.ucb(contexts), self.k).indices.cpu().numpy()
        self.model.update_confidence(contexts[arms])
        return arms

    def update(self, contexts, arms, base_rewards, super_reward):
        for arm, reward in zip(arms, base_rewards):
            self.model.add_sample_and_train(contexts[arm], reward)


class LinUCB:
    """Disjoint LinUCB (one ridge regression per base arm), playing the K arms with the highest UCB.

    Arms that were never played have A = I and b = 0, so only the played arms are stored.
    """

    def __init__(self, env, k, device, alpha):
        self.k, self.alpha, self.dim = k, alpha, env.dim
        self.slot = {}  # arm -> row of A_inv and b
        self.A_inv = np.empty((0, self.dim, self.dim))
        self.b = np.empty((0, self.dim))

    def select(self, contexts):
        ucb = self.alpha * np.sqrt(np.sum(contexts * contexts, axis=1))
        if self.slot:
            n = len(self.slot)
            arms = np.fromiter(self.slot, dtype=int, count=n)
            x, A_inv = contexts[arms], self.A_inv[:n]
            theta = np.einsum("nij,nj->ni", A_inv, self.b[:n])
            variance = np.einsum("ni,nij,nj->n", x, A_inv, x)
            ucb[arms] = np.sum(theta * x, axis=1) + self.alpha * np.sqrt(variance)
        return np.argpartition(ucb, -self.k)[-self.k:]

    def update(self, contexts, arms, base_rewards, super_reward):
        for arm, reward in zip(arms, base_rewards):
            i, x = self._row(arm), contexts[arm]
            A_inv_x = self.A_inv[i] @ x
            self.A_inv[i] -= np.outer(A_inv_x, A_inv_x) / (1 + x @ A_inv_x)  # Sherman-Morrison
            self.b[i] += reward * x

    def _row(self, arm):
        if arm not in self.slot:
            if len(self.slot) == len(self.A_inv):  # grow the storage geometrically
                extra = max(64, len(self.A_inv))
                self.A_inv = np.concatenate([self.A_inv, np.tile(np.eye(self.dim), (extra, 1, 1))])
                self.b = np.concatenate([self.b, np.zeros((extra, self.dim))])
            self.slot[arm] = len(self.slot)
        return self.slot[arm]


class NeuralMAB:
    """Greedy baseline: a linear reward model refit with Adam on all data, top-K predictions played."""

    def __init__(self, env, k, device, lr, epochs):
        self.k, self.lr, self.epochs, self.device = k, lr, epochs, device
        self.model = nn.Linear(env.dim, 1).to(device)
        self.inputs, self.targets = [], []

    def select(self, contexts):
        with torch.no_grad():
            x = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
            predicted = self.model(x).view(-1).cpu().numpy().astype(np.float64)
        return np.argsort(predicted)[-self.k:]

    def update(self, contexts, arms, base_rewards, super_reward):
        self.inputs.append(contexts[arms])
        self.targets.append(base_rewards)
        x = torch.as_tensor(np.concatenate(self.inputs), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.concatenate(self.targets), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            nn.functional.mse_loss(self.model(x), y.view(-1, 1)).backward()
            optimizer.step()


class CCMAB:
    """CC-MAB (Chen et al., 2018): uniform hypercube partition of the context space, explore
    under-explored hypercubes first and otherwise exploit the best average reward."""

    def __init__(self, env, k, device, rounds, rng):
        self.k, self.rng, self.dim = k, rng, env.dim
        self.cube_length = 1 / np.ceil(rounds ** (1 / (3 + self.dim)))
        self.t = 0
        self.plays, self.mean_reward = {}, {}

    def cubes(self, contexts):
        return [tuple(c) for c in (contexts / self.cube_length).astype(int)]

    def select(self, contexts):
        self.t += 1
        threshold = self.t ** (2 / (3 + self.dim)) * np.log(self.t)
        cubes = self.cubes(contexts)
        underexplored = np.array([self.plays.get(c, 0) <= threshold for c in cubes])
        explore = np.flatnonzero(underexplored)
        if len(explore) >= self.k:
            return self.rng.choice(explore, self.k, replace=False)
        exploit = np.flatnonzero(~underexplored)
        estimates = np.array([self.mean_reward.get(cubes[i], 0.0) for i in exploit])
        best = exploit[np.argsort(estimates)[len(explore) - self.k:]]
        return np.concatenate([explore, best])

    def update(self, contexts, arms, base_rewards, super_reward):
        for cube, reward in zip(self.cubes(contexts[arms]), base_rewards):
            n = self.plays[cube] = self.plays.get(cube, 0) + 1
            self.mean_reward[cube] = (self.mean_reward.get(cube, 0.0) * (n - 1) + reward) / n
