"""
Bandit.py — Core classes and visualization for the multi-armed bandit assignment.

This module defines:
  • An abstract Bandit interface.
  • Two concrete algorithms:
      - EpsilonGreedy (Gaussian rewards, ε_t = ε0 / t decay)
      - ThompsonSampling (Gaussian rewards, Normal–Normal conjugacy)
  • A Visualization helper to plot learning curves and comparisons.
  • Runner utilities: comparison(), summarize_results(), run_experiment().

Configuration
-------------
Runtime parameters are read from the external `config` module:
  - Bandit_Reward: list[float] of true means per arm (e.g., [1, 2, 3, 4])
  - NumberOfTrials: int horizon T
  - Random_Seed: base RNG seed (int)
  - Output_CSV: str path for results CSV
  - Algo_Epsilon_name: name label for ε-greedy in plots/CSV
  - Algo_TS_name: name label for Thompson Sampling in plots/CSV
  - Show_Learning_Plots: bool to toggle learning plots
  - Show_Comparison_Plots: bool to toggle comparison plots

This file assumes (by your latest config):
  - Epsilon0, Sigma, TS_Mu0, TS_Tau0, and optionally Random_Seed_TS
    are defined in `config` for algorithm hyperparameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

import config


# =======================================================================
# Abstract Interface
# =======================================================================

class Bandit(ABC):
    """Abstract base class for a bandit algorithm.

    Subclasses must implement the standard MAB lifecycle:
    initialization, arm selection (pull), parameter update, experiment loop,
    and reporting.

    Attributes:
        p (list[float]): True arm means (environment parameters).
    """

    @abstractmethod
    def __init__(self, p: Iterable[float]) -> None:
        """Initialize the bandit with true arm means.

        Args:
            p: Iterable of true means (one per arm).
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:  # pragma: no cover - simple repr
        """Return a concise string representation."""
        raise NotImplementedError

    @abstractmethod
    def pull(self) -> int:
        """Select and return the next arm index to pull."""
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """Update internal estimates using the last (arm, reward) observation."""
        raise NotImplementedError

    @abstractmethod
    def experiment(self) -> None:
        """Run the full experiment for the configured number of trials."""
        raise NotImplementedError

    @abstractmethod
    def report(self) -> None:
        """Persist per-trial results and print summary statistics.

        Expected side effects:
            - Append rows to the CSV at `config.Output_CSV`.
            - Log cumulative/average reward and cumulative regret.
        """
        raise NotImplementedError


# =======================================================================
# Visualization
# =======================================================================

class Visualization:
    """Plot learning dynamics and algorithm comparisons.

    Provided figures:
        • plot1: Running-average reward (learning) on linear and log scales.
        • plot2: Cumulative reward and cumulative regret comparisons.

    Notes:
        - Plots are saved as PNGs; showing is also invoked for interactive runs.
        - Plot generation is toggled via config.Show_Learning_Plots and
          config.Show_Comparison_Plots.
    """

    # Utilities

    @staticmethod
    def _moving_avg(x: Iterable[float], w: int) -> np.ndarray:
        """Compute a safe moving average with automatic window shrink.

        If the sequence is shorter than the requested window, a smaller
        effective window is used (down to 1).

        Args:
            x: 1-D data.
            w: Window size for the moving average.

        Returns:
            np.ndarray: Moving-average series (valid-length convolution).
        """
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return np.array([])
        if w <= 0:
            w = 1
        if len(arr) < w:
            # shrink window but keep it ≥1
            w = max(1, len(arr) // 10 or 1)
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    @staticmethod
    def _ensure_1d(a: Optional[Iterable[float]], name: str) -> np.ndarray:
        """Ensure the input is a 1-D array.

        Args:
            a: Sequence to validate.
            name: Variable name for error messages.

        Returns:
            np.ndarray: 1-D float array.

        Raises:
            ValueError: If `a` is None.
        """
        if a is None:
            raise ValueError(f"{name} cannot be None.")
        return np.asarray(a, dtype=float).ravel()

    # ---------------------------- plotters ---------------------------- #

    def plot1(
        self,
        eg_rewards: Iterable[float],
        ts_rewards: Iterable[float],
        window: int = 500,
        save_prefix: str = "plot_learning",
    ) -> None:
        """Learning curves: running-average rewards for both algorithms.

        Produces:
            - {save_prefix}_linear.png
            - {save_prefix}_log.png

        Args:
            eg_rewards: Per-trial rewards for ε-greedy.
            ts_rewards: Per-trial rewards for Thompson Sampling.
            window: Moving-average window size.
            save_prefix: File prefix for saved figures.
        """
        if not getattr(config, "Show_Learning_Plots", True):
            logger.info("plot1() skipped (Show_Learning_Plots=False).")
            return

        eg_rewards = self._ensure_1d(eg_rewards, "eg_rewards")
        ts_rewards = self._ensure_1d(ts_rewards, "ts_rewards")

        eg_ma = self._moving_avg(eg_rewards, window)
        ts_ma = self._moving_avg(ts_rewards, window)

        # ---- Linear scale ----
        plt.figure(figsize=(12, 6))
        plt.plot(
            eg_ma, label=f"{config.Algo_Epsilon_name} (running avg)", linestyle="--"
        )
        plt.plot(
            ts_ma, label=f"{config.Algo_TS_name} (running avg)", linestyle="-"
        )
        plt.xlabel("Trials")
        plt.ylabel(f"Running Avg Reward (window={window})")
        plt.title("Learning Process (Linear Scale)")
        plt.legend()
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        fname_lin = f"{save_prefix}_linear.png"
        plt.savefig(fname_lin, dpi=300)
        plt.show()
        logger.info(f"Saved: {fname_lin}")

        # Log scale (shift up if ≤ 0)
        min_val = min(np.min(eg_ma) if eg_ma.size else 0.0,
                      np.min(ts_ma) if ts_ma.size else 0.0)
        shift = (1.0 - min_val) if min_val <= 0 else 0.0
        eg_ma_log = eg_ma + shift
        ts_ma_log = ts_ma + shift

        plt.figure(figsize=(12, 6))
        plt.plot(
            eg_ma_log,
            label=f"{config.Algo_Epsilon_name} (running avg, shifted)",
            linestyle="--",
        )
        plt.plot(
            ts_ma_log,
            label=f"{config.Algo_TS_name} (running avg, shifted)",
            linestyle="-",
        )
        plt.yscale("log")
        plt.xlabel("Trials")
        plt.ylabel(
            f"Running Avg Reward (log scale){' + shift' if shift else ''}"
        )
        plt.title("Learning Process (Log Scale)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.35)
        plt.tight_layout()
        fname_log = f"{save_prefix}_log.png"
        plt.savefig(fname_log, dpi=300)
        plt.show()
        logger.info(f"Saved: {fname_log}")

    def plot2(
        self,
        eg_rewards: Iterable[float],
        ts_rewards: Iterable[float],
        eg_regrets: Iterable[float],
        ts_regrets: Iterable[float],
        save_prefix: str = "plot",
    ) -> None:
        """Comparison curves: cumulative reward and cumulative regret.

        Produces:
            - {save_prefix}_rewards.png
            - {save_prefix}_regrets.png

        Args:
            eg_rewards: Per-trial rewards for ε-greedy.
            ts_rewards: Per-trial rewards for Thompson Sampling.
            eg_regrets: Per-trial regrets for ε-greedy.
            ts_regrets: Per-trial regrets for Thompson Sampling.
            save_prefix: File prefix for saved figures.
        """
        if not getattr(config, "Show_Comparison_Plots", True):
            logger.info("plot2() skipped (Show_Comparison_Plots=False).")
            return

        eg_rewards = self._ensure_1d(eg_rewards, "eg_rewards")
        ts_rewards = self._ensure_1d(ts_rewards, "ts_rewards")
        eg_regrets = self._ensure_1d(eg_regrets, "eg_regrets")
        ts_regrets = self._ensure_1d(ts_regrets, "ts_regrets")

        # Cumulative Reward
        eg_cum = np.cumsum(eg_rewards)
        ts_cum = np.cumsum(ts_rewards)

        plt.figure(figsize=(12, 6))
        plt.plot(eg_cum, label=config.Algo_Epsilon_name, linestyle="--")
        plt.plot(ts_cum, label=config.Algo_TS_name, linestyle="-")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Comparison")
        plt.legend()
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        fname_rew = f"{save_prefix}_rewards.png"
        plt.savefig(fname_rew, dpi=300)
        plt.show()
        logger.info(f"Saved: {fname_rew}")

        # Cumulative Regret
        eg_cum_reg = np.cumsum(eg_regrets)
        ts_cum_reg = np.cumsum(ts_regrets)

        plt.figure(figsize=(12, 6))
        plt.plot(eg_cum_reg, label=config.Algo_Epsilon_name, linestyle="--")
        plt.plot(ts_cum_reg, label=config.Algo_TS_name, linestyle="-")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret Comparison")
        plt.legend()
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        fname_reg = f"{save_prefix}_regrets.png"
        plt.savefig(fname_reg, dpi=300)
        plt.show()
        logger.info(f"Saved: {fname_reg}")


# =======================================================================
# Algorithms
# =======================================================================

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy policy with 1/t decay on ε and Gaussian rewards.

    Reward model:
        R_t(a) ~ Normal(loc=p[a], scale=sigma)

    Exploration schedule:
        ε_t = ε0 / max(1, t)
    """

    def __init__(
        self,
        p: Iterable[float],
        epsilon0: float = 0.1,
        sigma: float = 1.0,
        trials: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Construct an Epsilon-Greedy agent.

        Args:
            p: True means per arm.
            epsilon0: Initial exploration weight.
            sigma: Observation standard deviation of rewards.
            trials: Number of trials to run (defaults to config.NumberOfTrials).
            rng_seed: RNG seed for reproducibility (defaults to config.Random_Seed).
        """
        # (Do not call super().__init__ because ABC is intentionally bare.)
        self.p: List[float] = list(p)
        self.n: int = len(self.p)
        self.trials: int = config.NumberOfTrials if trials is None else int(trials)

        # Policy hyperparameters
        self.epsilon0: float = float(epsilon0)
        self.sigma: float = float(sigma)
        self.t: int = 0  # time step counter

        # Estimates
        self.counts: List[int] = [0] * self.n
        self.values: List[float] = [0.0] * self.n  # sample means

        # Histories
        self.history_arms: List[int] = []
        self.history_rewards: List[float] = []
        self.per_trial_regrets: List[float] = []

        # For update()
        self._last_arm: Optional[int] = None
        self._last_reward: Optional[float] = None

        # Misc
        self.best_mean: float = max(self.p)
        self.name: str = config.Algo_Epsilon_name

        # RNG
        seed = config.Random_Seed if rng_seed is None else rng_seed
        random.seed(seed)
        np.random.seed(seed)

        logger.info(
            f"[{self.name}] init: epsilon0={self.epsilon0}, sigma={self.sigma}, n_arms={self.n}"
        )

    def __repr__(self) -> str:
        return f"EpsilonGreedy(epsilon0={self.epsilon0}, sigma={self.sigma}, n_arms={self.n})"

    def _current_epsilon(self) -> float:
        """Return ε_t = ε0 / t with safe lower bound on t and cap at 1.0."""
        t = max(1, self.t)
        return min(1.0, self.epsilon0 / t)

    @staticmethod
    def _argmax_with_random_tie_break(arr: Iterable[float]) -> int:
        """Argmax with uniform random tie-breaking.

        Args:
            arr: Iterable of numeric values.

        Returns:
            Index of one of the maximal elements, chosen uniformly at random.
        """
        a = np.asarray(arr, dtype=float)
        m = np.max(a)
        idx = np.flatnonzero(a == m)
        return int(np.random.choice(idx))

    def pull(self) -> int:
        """Choose an arm according to ε-greedy with 1/t decay."""
        self.t += 1
        eps = self._current_epsilon()
        if np.random.rand() < eps:
            # Explore
            return int(np.random.randint(0, self.n))
        # Exploit
        return self._argmax_with_random_tie_break(self.values)

    def update(self) -> None:
        """Incrementally update the sample-mean estimate for the last arm."""
        if self._last_arm is None:
            return
        arm = self._last_arm
        reward = float(self._last_reward)  # type: ignore[arg-type]
        self.counts[arm] += 1
        n = self.counts[arm]
        q_old = self.values[arm]
        self.values[arm] = q_old + (reward - q_old) / n

    def experiment(self) -> None:
        """Run the full ε-greedy experiment for `self.trials` steps."""
        logger.info(f"[{self.name}] starting experiment for {self.trials} trials.")
        for _ in range(self.trials):
            arm = self.pull()
            reward = float(np.random.normal(loc=self.p[arm], scale=self.sigma))

            # Log histories
            self.history_arms.append(arm)
            self.history_rewards.append(reward)
            self.per_trial_regrets.append(float(self.best_mean - self.p[arm]))

            # Update
            self._last_arm = arm
            self._last_reward = reward
            self.update()

        logger.info(f"[{self.name}] experiment finished.")

    def report(self) -> None:
        """Append results to CSV and log overall statistics.

        CSV columns:
            Bandit (int arm index), Reward (float), Algorithm (str label).
        """
        cum_reward = float(np.sum(self.history_rewards))
        cum_regret = float(np.sum(self.per_trial_regrets))
        avg_reward = float(np.mean(self.history_rewards)) if self.history_rewards else 0.0

        logger.info(f"[{self.name}] Cumulative Reward = {cum_reward:.4f}")
        logger.info(f"[{self.name}] Cumulative Regret = {cum_regret:.4f}")
        logger.info(f"[{self.name}] Average Reward per trial = {avg_reward:.4f}")

        df = pd.DataFrame(
            {
                "Bandit": self.history_arms,
                "Reward": self.history_rewards,
                "Algorithm": [self.name] * len(self.history_rewards),
            }
        )

        out = getattr(config, "Output_CSV", "results.csv")
        header_needed = not os.path.exists(out)
        df.to_csv(out, mode="a", index=False, header=header_needed)
        logger.info(f"[{self.name}] wrote {len(df)} rows to {out}")


class ThompsonSampling(Bandit):
    """Thompson Sampling with Gaussian rewards and known variance.

    Conjugate model (Normal–Normal):
        x | μ  ~ Normal(μ, σ²)  with known σ²
        μ      ~ Normal(μ0, 1/τ0)

    Posterior after n observations (sum s):
        τ' = τ0 + n * τ
        μ' = (τ0 * μ0 + τ * s) / τ''
    """

    def __init__(
        self,
        p: Iterable[float],
        mu0: float = 0.0,
        tau0: float = 1.0,
        sigma: float = 1.0,
        trials: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Construct a Thompson Sampling agent.

        Args:
            p: True arm means.
            mu0: Prior mean.
            tau0: Prior precision (1/variance).
            sigma: Known observation standard deviation.
            trials: Number of trials to run (defaults to config.NumberOfTrials).
            rng_seed: RNG seed (defaults to config.Random_Seed).
        """
        self.p: List[float] = list(p)
        self.n_arms: int = len(self.p)
        self.sigma: float = float(sigma)
        self.tau: float = 1.0 / (self.sigma**2)
        self.trials: int = config.NumberOfTrials if trials is None else int(trials)

        # Prior
        self.mu0: float = float(mu0)
        self.tau0: float = float(tau0)

        # Sufficient stats
        self.counts: List[int] = [0] * self.n_arms
        self.sums: List[float] = [0.0] * self.n_arms

        # Posterior params
        self.post_mu: List[float] = [self.mu0] * self.n_arms
        self.post_tau: List[float] = [self.tau0] * self.n_arms

        # Histories
        self.history_arms: List[int] = []
        self.history_rewards: List[float] = []
        self.per_trial_regrets: List[float] = []

        # For update()
        self._last_arm: Optional[int] = None
        self._last_reward: Optional[float] = None

        self.best_mean: float = max(self.p)
        self.name: str = config.Algo_TS_name

        # RNG
        seed = config.Random_Seed if rng_seed is None else rng_seed
        random.seed(seed)
        np.random.seed(seed)

        logger.info(
            f"[{self.name}] init: mu0={self.mu0}, tau0={self.tau0}, sigma={self.sigma}, arms={self.n_arms}"
        )

    def __repr__(self) -> str:
        return f"ThompsonSampling(mu0={self.mu0}, tau0={self.tau0}, sigma={self.sigma}, n_arms={self.n_arms})"

    def _refresh_posterior_for_arm(self, k: int) -> None:
        """Recompute posterior parameters for arm k."""
        n_k = self.counts[k]
        s_k = self.sums[k]
        tau_k = self.tau0 + n_k * self.tau
        mu_k = (self.tau0 * self.mu0 + self.tau * s_k) / max(tau_k, 1e-12)
        self.post_tau[k] = tau_k
        self.post_mu[k] = mu_k

    def pull(self) -> int:
        """Sample μ from each arm posterior and choose the argmax."""
        draws: List[float] = []
        for k in range(self.n_arms):
            var_k = 1.0 / max(self.post_tau[k], 1e-12)
            draws.append(float(np.random.normal(self.post_mu[k], np.sqrt(var_k))))
        a = np.asarray(draws)
        m = np.max(a)
        ties = np.flatnonzero(a == m)
        return int(np.random.choice(ties))

    def update(self) -> None:
        """Update sufficient stats and posterior for the last chosen arm."""
        if self._last_arm is None:
            return
        k = self._last_arm
        x = float(self._last_reward)  # type: ignore[arg-type]
        self.counts[k] += 1
        self.sums[k] += x
        self._refresh_posterior_for_arm(k)

    def experiment(self) -> None:
        """Run the full Thompson Sampling experiment for `self.trials` steps."""
        logger.info(f"[{self.name}] starting experiment for {self.trials} trials.")
        for _ in range(self.trials):
            arm = self.pull()
            reward = float(np.random.normal(loc=self.p[arm], scale=self.sigma))

            self.history_arms.append(arm)
            self.history_rewards.append(reward)
            self.per_trial_regrets.append(float(self.best_mean - self.p[arm]))

            self._last_arm = arm
            self._last_reward = reward
            self.update()
        logger.info(f"[{self.name}] experiment finished.")

    def report(self) -> None:
        """Append results to CSV and log overall statistics (see EG.report)."""
        cum_reward = float(np.sum(self.history_rewards))
        cum_regret = float(np.sum(self.per_trial_regrets))
        avg_reward = float(np.mean(self.history_rewards)) if self.history_rewards else 0.0

        logger.info(f"[{self.name}] Cumulative Reward = {cum_reward:.4f}")
        logger.info(f"[{self.name}] Cumulative Regret = {cum_regret:.4f}")
        logger.info(f"[{self.name}] Average Reward per trial = {avg_reward:.4f}")

        df = pd.DataFrame(
            {
                "Bandit": self.history_arms,
                "Reward": self.history_rewards,
                "Algorithm": [self.name] * len(self.history_rewards),
            }
        )
        out = getattr(config, "Output_CSV", "results.csv")
        df.to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        logger.info(f"[{self.name}] wrote {len(df)} rows to {out}")


# =======================================================================
# Runners / Reporting helpers
# =======================================================================

def comparison(eg: EpsilonGreedy, ts: ThompsonSampling, window: int = 500, save_prefix: str = "comparison") -> None:
    """Generate learning and comparison plots for two agents.

    Args:
        eg: Trained EpsilonGreedy instance (with histories).
        ts: Trained ThompsonSampling instance (with histories).
        window: Moving-average window for learning curves.
        save_prefix: Prefix for comparison plots output files.
    """
    viz = Visualization()

    # Learning process (running average reward, linear + log)
    viz.plot1(
        eg_rewards=eg.history_rewards,
        ts_rewards=ts.history_rewards,
        window=window,
        save_prefix="learning",
    )

    # Cumulative reward & cumulative regret
    viz.plot2(
        eg_rewards=eg.history_rewards,
        ts_rewards=ts.history_rewards,
        eg_regrets=eg.per_trial_regrets,
        ts_regrets=ts.per_trial_regrets,
        save_prefix=save_prefix,
    )

    eg_cum_r = float(np.sum(eg.history_rewards))
    eg_cum_g = float(np.sum(eg.per_trial_regrets))
    ts_cum_r = float(np.sum(ts.history_rewards))
    ts_cum_g = float(np.sum(ts.per_trial_regrets))
    logger.info(
        f"[SUMMARY] EG: reward={eg_cum_r:.3f}, regret={eg_cum_g:.3f} | "
        f"TS: reward={ts_cum_r:.3f}, regret={ts_cum_g:.3f}"
    )


def summarize_results(csv_path: str = "results.csv") -> pd.DataFrame:
    """Summarize per-algorithm totals from a results CSV.

    The CSV must contain columns: {Bandit, Reward, Algorithm}.

    For regret, we map each chosen arm index to its true mean using
    `config.Bandit_Reward` and compute per-trial regret as:
        regret_t = max(config.Bandit_Reward) − mean(chosen_arm_t)

    Args:
        csv_path: Path to results CSV file.

    Returns:
        pd.DataFrame: Aggregated summary with
            Trials, CumReward, AvgReward, CumRegret, AvgRegret (by Algorithm).

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(csv_path)
    required = {"Bandit", "Reward", "Algorithm"}
    if not required.issubset(df.columns):
        raise ValueError("results.csv must have columns: Bandit, Reward, Algorithm")

    means = list(config.Bandit_Reward)
    best_mean = max(means)
    df["ArmMean"] = df["Bandit"].astype(int).map(lambda i: means[i])
    df["Regret"] = best_mean - df["ArmMean"]

    summary = (
        df.groupby("Algorithm", as_index=False)
        .agg(
            Trials=("Reward", "size"),
            CumReward=("Reward", "sum"),
            AvgReward=("Reward", "mean"),
            CumRegret=("Regret", "sum"),
            AvgRegret=("Regret", "mean"),
        )
        .sort_values(["CumRegret", "Algorithm"])
    )

    logger.info("\n" + summary.to_string(index=False))
    return summary


def run_experiment(trials: Optional[int] = None) -> tuple[EpsilonGreedy, ThompsonSampling]:
    """Instantiate, train, and report both algorithms, then plot & summarize.

    Args:
        trials: Horizon for this run. If None, uses `config.NumberOfTrials`.

    Returns:
        A tuple (eg, ts) of trained agents.
    """
    if trials is None:
        trials = config.NumberOfTrials

    # Instantiate algorithms using values from `config`
    eg = EpsilonGreedy(
        p=config.Bandit_Reward,
        epsilon0=getattr(config, "Epsilon0", 0.1),
        sigma=getattr(config, "Sigma", 1.0),
        trials=trials,
        rng_seed=getattr(config, "Random_Seed", 42),
    )
    ts = ThompsonSampling(
        p=config.Bandit_Reward,
        mu0=getattr(config, "TS_Mu0", 0.0),
        tau0=getattr(config, "TS_Tau0", 1.0),
        sigma=getattr(config, "Sigma", 1.0),
        trials=trials,
        rng_seed=getattr(config, "Random_Seed_TS", getattr(config, "Random_Seed", 42) + 1),
    )

    # Run experiments
    eg.experiment()
    ts.experiment()

    # Persist and print stats
    eg.report()
    ts.report()

    # Plots (learning + cumulative comparisons)
    comparison(eg, ts, window=500, save_prefix="comparison")

    # Optional: table in logs
    summarize_results(getattr(config, "Output_CSV", "results.csv"))

    return eg, ts


# =======================================================================
# CLI entry
# =======================================================================

if __name__ == "__main__":
    # Example log levels—handy for graders reviewing console output.
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # Clean start for artifacts (optional; uncomment if your rubric requires)
    # out_csv = getattr(config, "Output_CSV", "results.csv")
    # if os.path.exists(out_csv):
    #     os.remove(out_csv)
    #     logger.info(f"Removed old CSV: {out_csv}")
    # for f in ["learning_linear.png", "learning_log.png",
    #           "comparison_rewards.png", "comparison_regrets.png"]:
    #     try:
    #         os.remove(f)
    #         logger.info(f"Removed old plot: {f}")
    #     except FileNotFoundError:
    #         pass

    # Run using config.NumberOfTrials by default
    run_experiment(trials=None)
