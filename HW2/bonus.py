#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bonus.py — Multi-armed bandit bonus implementation and evaluation.

This script implements and evaluates several bandit algorithms on a stationary
Gaussian multi-armed bandit environment:

Algorithms
----------
- EpsilonGreedy (ε_t = ε0 / t decay)
- ThompsonSampling (Gaussian likelihood with known variance)
- UCB1 (Gaussian variant: Q_i + c * sqrt(2 ln t / n_i))
- Hybrid_TS2Greedy (TS warm-start for a fraction of the horizon, then
  greedy exploitation using frozen posterior means)

Evaluation
----------
Runs each algorithm on the same environment across multiple RNG seeds,
reports noise-robust metrics (including *de-noised* cumulative reward based on
true means), and saves summary plots and a CSV with raw results. Prior outputs
(CSV/plots) are cleared at the start of each run so results are reproducible
and easy to re-generate.

Usage
-----
    python bonus.py

Notes
-----
- Configuration lives in `config_bonus.py` and is imported as `CFG`.
- “De-noised” reward is computed from chosen arms and *true* arm means; it
  removes Gaussian observation noise and is robust for comparing policies.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# External configuration (edit values in config_bonus.py, not here)
# ---------------------------------------------------------------------------
import config_bonus as CFG


def clear_outputs() -> None:
    """Remove previous CSV and plots so every run starts clean.

    This helper deletes:
      * The results CSV specified by ``CFG.OUT_CSV`` if it exists.
      * Any PNG plots whose filenames start with ``CFG.PLOT_PREFIX``.
    """
    # Remove CSV
    if os.path.exists(CFG.OUT_CSV):
        try:
            os.remove(CFG.OUT_CSV)
            print(f"[CLEAN] Removed old CSV: {CFG.OUT_CSV}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Could not remove {CFG.OUT_CSV}: {e}")

    # Remove plots matching the prefix (e.g., bonus_*.png)
    patterns = [f"{CFG.PLOT_PREFIX}_*.png"]
    removed = 0
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.remove(path)
                removed += 1
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Could not remove {path}: {e}")
    if removed:
        print(f"[CLEAN] Removed {removed} old plot(s) with prefix '{CFG.PLOT_PREFIX}_*.png'")


# ===========================================================================
# Environment
# ===========================================================================

@dataclass
class Env:
    """Stationary Gaussian bandit environment.

    Rewards from arm ``a`` are drawn as ``Normal(means[a], sigma^2)``.

    Attributes:
        means: True means for each arm (stationary).
        sigma: Known standard deviation of Gaussian rewards.
    """
    means: Sequence[float]
    sigma: float

    def pull_reward(self, arm: int, rng: np.random.Generator) -> float:
        """Sample a reward for the specified arm.

        Args:
            arm: Index of the arm to pull.
            rng: Numpy random generator to use.

        Returns:
            A single reward draw as ``float``.
        """
        return float(rng.normal(loc=self.means[arm], scale=self.sigma))

    @property
    def best_mean(self) -> float:
        """Best (maximum) true mean across all arms."""
        return float(max(self.means))


# ===========================================================================
# Base Policy
# ===========================================================================

class Policy:
    """Abstract policy with a minimal API.

    Subclasses must implement :meth:`select_arm` and :meth:`update`.

    Attributes:
        n: Number of arms.
        name: Algorithm name (used in logs/plots/CSV).
        t: Current time step (1-based inside :meth:`run` loop).
        history_arms: Chosen arm index at each step.
        history_rewards: Observed reward at each step (with noise).
        per_trial_regrets: Instantaneous regret at each step computed as
            ``best_mean - means[chosen_arm]`` (noise-free regret).
    """

    def __init__(self, n_arms: int, name: str):
        """Initialize a policy shell (no environment attached yet)."""
        self.n = int(n_arms)
        self.name = name
        self.t = 0
        self.history_arms: List[int] = []
        self.history_rewards: List[float] = []
        self.per_trial_regrets: List[float] = []

    def select_arm(self, rng: np.random.Generator) -> int:
        """Return the index of the next arm to pull.

        Must be implemented by subclasses.

        Args:
            rng: Random generator for stochastic choices.

        Returns:
            Arm index to pull.
        """
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        """Update internal state from the observed transition.

        Must be implemented by subclasses.

        Args:
            arm: Arm index that was pulled.
            reward: Observed reward for that arm.
        """
        raise NotImplementedError

    def run(self, env: Env, T: int, rng: np.random.Generator) -> None:
        """Execute ``T`` steps in the given environment.

        This method resets histories, then repeatedly:
          1) selects an arm via :meth:`select_arm`,
          2) samples a reward from the environment,
          3) calls :meth:`update` with (arm, reward),
          4) records histories and instantaneous regret.

        Args:
            env: Bandit environment.
            T: Horizon (number of time steps).
            rng: Random generator for the rollout.
        """
        # Reset per-run state
        self.t = 0
        self.history_arms.clear()
        self.history_rewards.clear()
        self.per_trial_regrets.clear()

        best = env.best_mean
        for _ in range(T):
            self.t += 1
            a = self.select_arm(rng)
            r = env.pull_reward(a, rng)
            self.update(a, r)
            self.history_arms.append(a)
            self.history_rewards.append(r)
            # Regret is computed on true means (noise-free)
            self.per_trial_regrets.append(best - env.means[a])


# ===========================================================================
# Policies
# ===========================================================================

class EpsilonGreedy(Policy):
    """Epsilon-Greedy with harmonic decay: ``ε_t = ε0 / t``.

    With probability ``ε_t`` the policy explores uniformly at random; otherwise,
    it exploits the arm with the largest sample-mean estimate (random tie-break).
    """

    def __init__(self, n_arms: int, epsilon0: float = 0.1):
        """Construct an Epsilon-Greedy policy.

        Args:
            n_arms: Number of arms in the bandit.
            epsilon0: Initial exploration weight (decays as 1/t).
        """
        super().__init__(n_arms, name="EpsilonGreedy")
        self.epsilon0 = float(epsilon0)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def _eps(self) -> float:
        """Current epsilon value ``ε_t``."""
        t = max(1, self.t)
        return self.epsilon0 / t

    def select_arm(self, rng: np.random.Generator) -> int:
        """Select an arm via ε-greedy with random tie-breaking."""
        if rng.random() < self._eps():
            # Explore
            return int(rng.integers(self.n))
        # Exploit — break ties at random among maxima
        m = self.values.max()
        idx = np.flatnonzero(self.values == m)
        return int(rng.choice(idx))

    def update(self, arm: int, reward: float) -> None:
        """Incremental mean update for the selected arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        q = self.values[arm]
        self.values[arm] = q + (reward - q) / n


class ThompsonSamplingGaussian(Policy):
    """Thompson Sampling for Gaussian rewards with known variance.

    Likelihood:  x | μ  ~ Normal(μ, σ²)  (σ known; τ = 1/σ²)
    Prior:       μ     ~ Normal(μ0, 1/τ0)

    Posterior after n observations with sum s:
        τ' = τ0 + n τ
        μ' = (τ0 μ0 + τ s) / τ''
    """

    def __init__(self, n_arms: int, mu0: float = 0.0, tau0: float = 1.0, sigma: float = 1.0):
        """Construct a Gaussian-TS policy with Normal-Normal conjugacy.

        Args:
            n_arms: Number of arms.
            mu0: Prior mean for each arm.
            tau0: Prior precision (1/variance) for each arm.
            sigma: Known observation std of rewards (shared across arms).
        """
        super().__init__(n_arms, name="ThompsonSampling")
        # Likelihood precision
        self.tau = 1.0 / (sigma**2)
        # Normal prior N(mu0, 1/tau0)
        self.mu0 = float(mu0)
        self.tau0 = float(tau0)
        # Posterior parameters (one pair per arm)
        self.post_mu = np.full(n_arms, self.mu0, dtype=float)
        self.post_tau = np.full(n_arms, self.tau0, dtype=float)
        # Sufficient statistics
        self.counts = np.zeros(n_arms, dtype=int)
        self.sums = np.zeros(n_arms, dtype=float)

    def _refresh(self, k: int) -> None:
        """Recompute posterior params for arm ``k``."""
        tau_k = self.tau0 + self.counts[k] * self.tau
        mu_k = (self.tau0 * self.mu0 + self.tau * self.sums[k]) / max(tau_k, 1e-12)
        self.post_tau[k] = tau_k
        self.post_mu[k] = mu_k

    def select_arm(self, rng: np.random.Generator) -> int:
        """Sample a candidate mean from each posterior and pick the argmax."""
        draws = rng.normal(self.post_mu, np.sqrt(1.0 / np.maximum(self.post_tau, 1e-12)))
        m = draws.max()
        idx = np.flatnonzero(draws == m)
        return int(rng.choice(idx))

    def update(self, arm: int, reward: float) -> None:
        """Update sufficient stats and arm posterior after an observation."""
        self.counts[arm] += 1
        self.sums[arm] += reward
        self._refresh(arm)


class UCB1Gaussian(Policy):
    """Gaussian UCB1: ``Q_i + c * sqrt(2 ln t / n_i)``.

    Each arm is pulled once at the beginning; thereafter, the policy selects
    the arm maximizing the UCB index.
    """

    def __init__(self, n_arms: int, c: float = 1.0):
        """Construct a UCB1 policy.

        Args:
            n_arms: Number of arms.
            c: Exploration coefficient in the UCB index.
        """
        super().__init__(n_arms, name="UCB1")
        self.c = float(c)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self, rng: np.random.Generator) -> int:  # noqa: ARG002
        """Select an arm using the UCB1 index (with initial pulls)."""
        # Initial phase: pull each arm once
        for a in range(self.n):
            if self.counts[a] == 0:
                return a
        # UCB index
        ucb = self.values + self.c * np.sqrt(2.0 * math.log(self.t) / np.maximum(self.counts, 1))
        m = ucb.max()
        idx = np.flatnonzero(ucb == m)
        return int(np.random.choice(idx))

    def update(self, arm: int, reward: float) -> None:
        """Incremental mean update after observing reward."""
        self.counts[arm] += 1
        n = self.counts[arm]
        q = self.values[arm]
        self.values[arm] = q + (reward - q) / n


class HybridTS2Greedy(Policy):
    """Hybrid policy: TS warm-start then deterministic greedy.

    The policy first runs Thompson Sampling for a fraction of the horizon
    (``warm_frac``), then *freezes* the posterior means and exploits greedily
    based on those frozen values for the remainder of the run.
    """

    def __init__(
        self,
        n_arms: int,
        sigma: float = 1.0,
        mu0: float = 0.0,
        tau0: float = 1.0,
        warm_frac: float = 0.1,
    ):
        """Construct the TS→Greedy hybrid policy.

        Args:
            n_arms: Number of arms.
            sigma: Observation std used by the internal TS phase.
            mu0: Prior mean for TS.
            tau0: Prior precision for TS.
            warm_frac: Fraction of the horizon assigned to TS warm-start (0,1).
        """
        super().__init__(n_arms, name="Hybrid_TS2Greedy")
        self.ts = ThompsonSamplingGaussian(n_arms, mu0=mu0, tau0=tau0, sigma=sigma)
        self.greedy_values = np.zeros(n_arms, dtype=float)
        self.greedy_counts = np.zeros(n_arms, dtype=int)
        self.phase_switch_t: int | None = None
        self.warm_frac = float(warm_frac)

    def select_arm(self, rng: np.random.Generator) -> int:
        """Use TS during warm-start; after switch, exploit frozen means."""
        if self.phase_switch_t is None or self.t <= self.phase_switch_t:
            return self.ts.select_arm(rng)
        m = self.greedy_values.max()
        idx = np.flatnonzero(self.greedy_values == m)
        return int(rng.choice(idx))

    def update(self, arm: int, reward: float) -> None:
        """Update internal learner; freeze means exactly at the switch."""
        # Update TS during warm phase
        if self.phase_switch_t is None or self.t <= self.phase_switch_t:
            self.ts.update(arm, reward)
        else:
            # Not used for decision, but kept for completeness/logging symmetry
            self.greedy_counts[arm] += 1

        # Freeze posterior means the *moment* we cross into the greedy phase
        if self.phase_switch_t is not None and self.t == self.phase_switch_t:
            self.greedy_values = self.ts.post_mu.copy()

    def run(self, env: Env, T: int, rng: np.random.Generator) -> None:
        """Set the switching time and delegate to base runner."""
        # Ensure at least one step of TS
        self.phase_switch_t = max(1, int(self.warm_frac * T))
        super().run(env, T, rng)


# ===========================================================================
# Evaluation (multi-seed)
# ===========================================================================

def denoised_cum_reward(arms: Sequence[int], means: Sequence[float]) -> float:
    """Compute *de-noised* cumulative reward implied by chosen arms.

    Instead of summing noisy realizations, we count how many times each arm
    was pulled and multiply by the true mean of that arm.

    Args:
        arms: Sequence of chosen arm indices.
        means: True means (same order as arm indices).

    Returns:
        Sum_k (N_k * μ_k) as a float.
    """
    counts = np.bincount(np.asarray(arms, dtype=int), minlength=len(means))
    return float((counts * np.asarray(means)).sum())


def evaluate_once(env: Env, policy: Policy, T: int, seed: int) -> Dict[str, float]:
    """Run one policy once and collect key metrics.

    Args:
        env: Bandit environment.
        policy: Policy instance to evaluate.
        T: Horizon (time steps).
        seed: RNG seed for this rollout.

    Returns:
        Dictionary with algorithm name, seed, trials, cumulative regret,
        noisy cumulative reward, and de-noised cumulative reward.
    """
    rng = np.random.default_rng(seed)
    policy.run(env, T, rng)

    cum_regret = float(np.sum(policy.per_trial_regrets))
    reward_noisy = float(np.sum(policy.history_rewards))
    reward_denoised = denoised_cum_reward(policy.history_arms, env.means)

    return dict(
        Algorithm=policy.name,
        Seed=seed,
        Trials=T,
        CumRegret=cum_regret,
        CumRewardNoisy=reward_noisy,
        CumRewardDenoised=reward_denoised,
    )


def evaluate_all(env: Env, trials: int, seeds: List[int]) -> pd.DataFrame:
    """Evaluate all configured algorithms over multiple seeds.

    Args:
        env: Bandit environment.
        trials: Horizon per run.
        seeds: List of RNG seeds to use (one run per seed per algorithm).

    Returns:
        DataFrame with one row per (algorithm × seed) run.
    """
    results: List[Dict[str, float]] = []
    for seed in seeds:
        # Fresh policy instances per seed (no state carry-over).
        algos: List[Policy] = [
            EpsilonGreedy(n_arms=len(env.means), epsilon0=CFG.EG_EPS0),
            ThompsonSamplingGaussian(
                n_arms=len(env.means),
                mu0=CFG.TS_MU0,
                tau0=CFG.TS_TAU0,
                sigma=CFG.SIGMA,
            ),
            UCB1Gaussian(n_arms=len(env.means), c=CFG.UCB1_C),
            HybridTS2Greedy(
                n_arms=len(env.means),
                sigma=CFG.SIGMA,
                mu0=CFG.TS_MU0,
                tau0=CFG.TS_TAU0,
                warm_frac=CFG.HYBRID_WARM_FRAC,
            ),
        ]
        for pol in algos:
            row = evaluate_once(env, pol, trials, seed)
            results.append(row)
    return pd.DataFrame(results)


# ===========================================================================
# Reporting & plots
# ===========================================================================

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by algorithm (mean ± std over seeds).

    Args:
        df: Long-form results with columns produced by :func:`evaluate_all`.

    Returns:
        Summary DataFrame sorted by ``MeanCumRegret`` ascending.
    """
    agg = (
        df.groupby("Algorithm", as_index=False)
        .agg(
            Trials=("Trials", "first"),
            Runs=("Seed", "count"),
            MeanCumRegret=("CumRegret", "mean"),
            StdCumRegret=("CumRegret", "std"),
            MeanDenReward=("CumRewardDenoised", "mean"),
            StdDenReward=("CumRewardDenoised", "std"),
            MeanNoisyReward=("CumRewardNoisy", "mean"),
            StdNoisyReward=("CumRewardNoisy", "std"),
        )
        .sort_values("MeanCumRegret")
    )
    return agg


def quick_side_by_side(df: pd.DataFrame) -> None:
    """Create simple comparative plots from the CSV aggregate.

    Two figures are produced:
      1) Mean cumulative regret (lower is better).
      2) Mean de-noised cumulative reward (higher is better).

    Args:
        df: Long-form results DataFrame (algorithm × seed).
    """
    # --- Regret (bars as a simple line for compactness) ---
    plt.figure(figsize=(9, 5))
    order = df.groupby("Algorithm")["CumRegret"].mean().sort_values(ascending=True)
    xs = range(len(order))
    plt.plot(xs, order.values, marker="o")
    plt.xticks(xs, order.index, rotation=10)
    plt.ylabel("Mean cumulative regret (over seeds)")
    plt.title("Bonus: Mean Cumulative Regret")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(f"{CFG.PLOT_PREFIX}_mean_cumulative_regret.png", dpi=300)
    plt.close()

    # --- De-noised reward (bars as a simple line) ---
    plt.figure(figsize=(9, 5))
    ord2 = df.groupby("Algorithm")["CumRewardDenoised"].mean().sort_values(ascending=False)
    xs = range(len(ord2))
    plt.plot(xs, ord2.values, marker="o")
    plt.xticks(xs, ord2.index, rotation=10)
    plt.ylabel("Mean de-noised cumulative reward")
    plt.title("Bonus: Mean De-noised Cumulative Reward")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(f"{CFG.PLOT_PREFIX}_mean_denoised_cumulative_reward.png", dpi=300)
    plt.close()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Entry point: clear outputs, run evaluation, print summary, and plot."""
    # Start fresh each run for clean, reproducible artifacts
    clear_outputs()

    # Build environment from external config
    env = Env(CFG.BANDIT_MEANS, CFG.SIGMA)

    # Evaluate and persist (header is guaranteed True now because we just cleared)
    df = evaluate_all(env, CFG.TRIALS, CFG.SEEDS)
    df.to_csv(CFG.OUT_CSV, mode="w", index=False, header=True)
    print(f"[OK] Wrote {len(df)} rows to {CFG.OUT_CSV}")

    # Summary table
    summ = summarize(df)
    print("\n=== Summary (sorted by MeanCumRegret ↓) ===")
    print(summ.to_string(index=False))

    # Winner by regret
    winner = summ.iloc[0]
    print(
        f"\nWinner (by mean cumulative regret over {int(winner.Runs)} seeds): "
        f"{winner.Algorithm} | MeanCumRegret={winner.MeanCumRegret:.2f}"
    )

    # Plots
    quick_side_by_side(df)
    print(
        f"[OK] Saved plots: "
        f"{CFG.PLOT_PREFIX}_mean_cumulative_regret.png, {CFG.PLOT_PREFIX}_mean_denoised_cumulative_reward.png"
    )


if __name__ == "__main__":
    main()
