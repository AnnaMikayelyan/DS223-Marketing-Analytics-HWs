# A/B Testing : Epsilon Greedy and Thompson Sampling

## Overview

This repository contains the implementation of a **multi-armed bandit experiment** using **Epsilon Greedy** and **Thompson Sampling** algorithms. The objective is to simulate advertisement testing (A/B Testing) by evaluating four different advertisement options (bandits). The experiment design includes the following:

1. **Epsilon Greedy**: Implements the classic epsilon-greedy approach with decay in epsilon over time.
2. **Thompson Sampling**: Uses Bayesian inference with known precision (Gaussian likelihood with conjugate priors).

### **Key Features**

- **Epsilon-Greedy Algorithm** with 1/t decay on epsilon.
- **Thompson Sampling** using Normal–Normal conjugacy for Gaussian rewards.
- **Visualization** of the learning process and cumulative rewards.
- **Results storage** in CSV format for analysis.
- **Reporting** of cumulative reward and regret.

---

## **Experiment Design**

### **Bandit Class**
A **Bandit** abstract class has been created to provide the structure for the two algorithms. Both **EpsilonGreedy** and **ThompsonSampling** inherit from the Bandit class, which includes the following abstract methods:
- `__init__(self, p: Iterable[float])`
- `__repr__(self) -> str`
- `pull(self) -> int`
- `update(self) -> None`
- `experiment(self) -> None`
- `report(self) -> None`

### **Epsilon-Greedy Algorithm**

The **Epsilon-Greedy** algorithm chooses arms based on a mix of exploration and exploitation. The exploration is controlled by a decaying epsilon value:
- Initially, epsilon (`ε0`) is set to **0.1**.
- The exploration factor decays as `ε_t = ε0 / t`, where `t` is the current time step.

#### **Key Aspects**:
1. **Decaying Exploration**: Reduces exploration over time by decaying epsilon.
2. **Exploration vs Exploitation**: Randomly explores arms with probability `ε_t` and exploits the best arm with probability `1 - ε_t`.

### **Thompson Sampling Algorithm**

**Thompson Sampling** uses a **Bayesian approach** for arm selection:
- **Prior**: Normal distribution with mean `μ0` and precision `τ0`.
- **Likelihood**: Gaussian with mean `μ` and known variance `σ^2`.
- **Posterior**: The posterior mean is updated based on the observations.

#### **Key Aspects**:
1. **Conjugate Model**: Normal likelihood with a Normal prior.
2. **Reward Model**: Gaussian rewards with known precision.

### **Experiment Configuration**

The configuration for the experiment is provided in `config.py`, where the key parameters such as the number of trials, epsilon, and precision values are defined.

- **Bandit_Reward**: The true mean reward for each arm is set to `[1, 2, 3, 4]`.
- **NumberOfTrials**: The number of trials (experiments) is set to **20000**.
- **Epsilon0**: Initial exploration value for the Epsilon-Greedy algorithm is **0.1**.
- **Sigma**: The standard deviation of Gaussian rewards is **1.0**.

The results are saved in CSV files for each algorithm with columns:
- `Bandit`: Arm index.
- `Reward`: Reward obtained from that arm.
- `Algorithm`: Name of the algorithm used (either "Epsilon Greedy" or "Thompson Sampling").

---

## **Implementation Steps**

1. **Create Bandit Class**:
    - Implement the abstract `Bandit` class with methods for initialization, arm selection, update, experiment, and reporting.

2. **Create EpsilonGreedy and ThompsonSampling Classes**:
    - Implement the **EpsilonGreedy** and **ThompsonSampling** classes inheriting from `Bandit`.
    - Ensure that `EpsilonGreedy` implements **decaying epsilon** and `ThompsonSampling` uses **Gaussian likelihood** with known precision.

3. **Design the Experiment**:
    - Simulate 20,000 trials for each algorithm, and evaluate their performance by storing cumulative rewards and regrets.

4. **Visualization**:
    - Use `matplotlib` to plot the **learning curves** (running average of rewards) for both algorithms.
    - Compare the **cumulative rewards** and **regrets** using side-by-side plots.

5. **Reporting**:
    - Log the **cumulative reward** and **cumulative regret** for both algorithms.
    - Save the results in CSV format (`results.csv`).

---

## **Running the Experiment**

To run the experiment, use the following command:

```bash
python bandit.py
```

## Bonus Files

### **bonus.py**

The `bonus.py` file implements additional experiments and algorithms for a **multi-armed bandit problem**, focusing on **multi-seed evaluation** and more advanced bandit strategies. The file includes:

#### **Algorithms Implemented**:

- **EpsilonGreedy** (ε_t = ε0 / t decay)
- **ThompsonSampling** (Gaussian likelihood with known variance)
- **UCB1** (Gaussian variant: `Q_i + c * sqrt(2 ln t / n_i)`)
- **Hybrid_TS2Greedy** (TS warm-start for a fraction of the horizon, then greedy exploitation using frozen posterior means)

#### **Evaluation**:

The script runs each algorithm on the same environment across multiple RNG seeds, reports **noise-robust metrics** (including de-noised cumulative reward based on true means), and saves summary plots and a CSV with raw results.

#### **Purpose**:

The **bonus script** allows for the comparison of different exploration strategies across multiple trials, including an analysis of the **cumulative regret** and **de-noised cumulative reward** for each algorithm.

---

### **config_bonus.py**

The `config_bonus.py` file contains configuration parameters specific to the `bonus.py` implementation. It allows easy modification of key settings like:

#### **Bandit Environment**:

- **BANDIT_MEANS**: The true mean rewards for each arm (e.g., `[1.0, 2.0, 3.0, 4.0]`).
- **SIGMA**: The standard deviation of the rewards (set to **1.0**).

#### **Experiment Horizon & Seeds**:

- **TRIALS**: Number of trials for each experiment (e.g., **20000**).
- **SEEDS**: List of RNG seeds used for multiple experiment runs (e.g., `[11, 23, 37, 41]`).

#### **Algorithm Hyperparameters**:

- **EG_EPS0**: Initial epsilon for **EpsilonGreedy** (**0.1**).
- **TS_MU0**, **TS_TAU0**: Prior parameters for **ThompsonSampling** (**0.0** and **1.0** respectively).
- **UCB1_C**: Exploration coefficient for **UCB1** (**1.0**).
- **HYBRID_WARM_FRAC**: Fraction of the horizon allocated to **ThompsonSampling** before switching to **Greedy** in **Hybrid_TS2Greedy**.

---

### **Dependencies**

To install the required libraries for this project, run:

```bash
pip install -r requirements.txt
```
