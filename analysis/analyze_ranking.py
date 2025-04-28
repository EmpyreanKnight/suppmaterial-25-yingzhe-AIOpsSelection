import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
#from scipy.stats import ranksums
import json
import kendall_w as kw
#from cliffs_delta import cliffs_delta

# targets: 
# Spearman's rho OR Kendall's tau for pair-wise similarity indicator
# Jaccard similarity (top k=3, 5, 10) for similarity
# between the model rankings of mechanisms and the oracle on each round each period 
# Dataset-Model-Scenario-Round-Wnd
# visualization: one line plot, color for different measurement

# Kendall's W for consistency
# among the 100 rounds, for each time period, box plot for visualization
# Dataset-Model-Scenario-Wnd

# Wilcoxon Rank Sum: significance of the performance difference 
# Cliff's delta: effect size
# or just scott-knott to grouping?

datasets = ['google', 'backblaze', 'alibaba']
models = ['lr', 'nn', 'rf', 'cart']
scenarios = ['crc', 'laf', 'temporal', 'temporal_rev', 'dist', 'dist_leak']
num_periods = {
    'google': 28,
    'backblaze': 36,
    'alibaba': 8
}
pd.set_option('display.max_colwidth', None)

output_header = ['Dataset', 'Model', 'Scenario', 'Testing Period', 'Round', 'Kendall Tau', 'Jaccard 3', 'Jaccard 5', 'Jaccard 10', 'Kendall W']
output_ls = []
output_file = r'ranking_analysis.csv'

for dataset in datasets:
    for model in models:
        df = pd.read_csv(f'results/selection_{dataset}_{model}.csv')
        for scenario in scenarios:
            for testing_period in range(num_periods[dataset]//2+2, num_periods[dataset]+1):
                print(f'Now running: {dataset}/{model}/{scenario}/{testing_period}')
                rankings = []
                for round in range(100):
                    oracle_ranking = df.query("Scenario == 'oracle' and Round == @round and `Testing Period` == @testing_period")['Model Ranking']
                    if len(oracle_ranking) != 1:
                        print(f'Warning: early stop {dataset}/{model} on round {round+1}.')
                        break
                    oracle_ranking = json.loads(oracle_ranking.to_string(index=False))

                    ranking = df.query("Scenario == @scenario and Round == @round and `Testing Period` == @testing_period")['Model Ranking']
                    ranking = json.loads(ranking.to_string(index=False))

                    kendall_tau = kendalltau(np.argsort(oracle_ranking), np.argsort(ranking)).statistic
                    jaccard_3 = len(set(oracle_ranking[:3]).intersection(ranking[:3]))/len(set(oracle_ranking[:3]).union(ranking[:3]))
                    jaccard_5 = len(set(oracle_ranking[:5]).intersection(ranking[:5]))/len(set(oracle_ranking[:5]).union(ranking[:5]))
                    jaccard_10 = len(set(oracle_ranking[:10]).intersection(ranking[:10]))/len(set(oracle_ranking[:10]).union(ranking[:10]))

                    output_ls.append([dataset, model, scenario, testing_period, round, kendall_tau, jaccard_3, jaccard_5, jaccard_10, -1])

                    rankings.append(np.argsort(ranking))

                if rankings != []:
                    kendall_w = kw.compute_w(np.transpose(rankings).tolist())
                    output_ls.append([dataset, model, scenario, testing_period, -1, -1, -1, -1, -1, kendall_w])

df = pd.DataFrame(output_ls, columns=output_header)
df.to_csv(output_file, index=False)
