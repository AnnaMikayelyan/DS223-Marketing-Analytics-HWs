# Place for constants so we don't hard-code numbers

Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000
Random_Seed = 42
Random_Seed_TS = Random_Seed + 1   # for de-correlating Thompson Sampling

Epsilon0 = 0.1
Sigma = 1.0
TS_Mu0 = 0.0
TS_Tau0 = 1.0

Output_CSV = "results.csv"

Algo_Epsilon_name = "Epsilon Greedy"
Algo_TS_name = "Thompson Sampling"

# plotting toggles (can disable while debugging)
Show_Learning_Plots = True
Show_Comparison_Plots = True

