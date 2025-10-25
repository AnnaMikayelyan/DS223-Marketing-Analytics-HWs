# ---- Environment ----
BANDIT_MEANS = [1.0, 2.0, 3.0, 4.0]
SIGMA = 1.0

# ---- Experiment horizon & seeds ----
TRIALS = 20_000
SEEDS = [11, 23, 37, 41, 53, 59, 61, 73, 89, 97]

# ---- Outputs ----
OUT_CSV = "bonus_results.csv"
PLOT_PREFIX = "bonus"

# ---- Algorithm hyperparameters (so you can tune without touching code) ----
# Epsilon-Greedy (this file keeps the classic ε0/t in bonus.py; these are passed in)
EG_EPS0 = 0.1

# Thompson Sampling (Gaussian)
TS_MU0 = 0.0
TS_TAU0 = 1.0

# UCB1 (Gaussian)
UCB1_C = 1.0

# Hybrid TS→Greedy
HYBRID_WARM_FRAC = 0.10
