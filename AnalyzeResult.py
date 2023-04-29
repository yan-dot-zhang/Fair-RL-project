import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import seaborn as sns


def read_csv(fnPattern):
    # find files with fileformat
    pattern = re.compile(fnPattern + '[0-9]+.csv')
    files = [f for f in os.listdir('results') if pattern.match(f)]
    dfs = []
    for f in files:
        tmp_df = pd.read_csv('results/' + f)
        tmp_df['step'] = tmp_df.index
        # if terminal_observation exists, drop it
        if 'terminal_observation' in tmp_df.columns:
            tmp_df = tmp_df.drop(columns=['terminal_observation'])
        dfs.append(tmp_df)
    # concat all dataframes
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by=['step'], inplace=True, ignore_index=True)
    return df

def compute_GGF(df, reward_n = 2, weight_coef = 10):
    omega = np.array([1 / (weight_coef ** i) for i in range(reward_n)])
    max_val = df[['Sea_Otters', 'Northern_Abalone']].max(axis=1)
    min_val = df[['Sea_Otters', 'Northern_Abalone']].min(axis=1)
    # compute GGF, dot product of omega and two values
    GGF = np.dot(omega, np.array([min_val, max_val]))
    return GGF


def read_result_data(pattern_list, keys_list=None):
    if keys_list is None:
        keys_list = pattern_list
    df_list = []
    for pattern, key in zip(pattern_list, keys_list):
        tmp_df = read_csv(pattern)
        tmp_df['GGF_Score'] = compute_GGF(tmp_df, reward_n=2, weight_coef=2)
        # subset of df_ppo
        tmp_df = tmp_df[tmp_df['step']%20 == 18]
        tmp_df = tmp_df[tmp_df['step'] < 60000]
        tmp_df['Algorithm'] = [key] * tmp_df.shape[0]
        df_list.append(tmp_df)
    # merge dataframes
    data_ = pd.concat(df_list, ignore_index=True)
    return data_

pattern_list  = ['reward_a2c', 'reward_a2cggi', 'reward_ppo', 'reward_ppoggi','reward_dqn', 'reward_dqnggi','reward_random']
keys_list = ['A2C', 'GGF-A2C', 'PPO', 'GGF-PPO', 'DQN', 'GGF-DQN', 'Random']
result_df = read_result_data(pattern_list, keys_list)

# learning figure
plt.figure(figsize=(14, 7))
sns.lineplot(data=result_df, x="step", y="Sum", hue='Algorithm')
plt.legend()
plt.show()

# boxplot with matplotlib
final_result_df = result_df[result_df['step'] == max(result_df['step'])]
boxplot_data = []
keys_list_boxplot = ['DQN', 'GGF-DQN', 'A2C', 'GGF-A2C', 'PPO', 'GGF-PPO']
for key in keys_list_boxplot:
    boxplot_data.append(final_result_df[final_result_df['Algorithm'] == key]['GGF_Score'].to_numpy())
# boxplot with matplotlib
plt.figure(figsize=(10, 5))
plt.boxplot(boxplot_data, labels=keys_list_boxplot)
plt.show()

# barchart
# for each algorithm, compute average density and std, use dataframe.groupby
final_result_barchart = final_result_df.groupby(['Algorithm']).mean().reset_index()
final_result_barchart['Sea_Otters_std'] = final_result_df.groupby(['Algorithm']).std().reset_index()['Sea_Otters']
final_result_barchart['Northern_Abalone_std'] = final_result_df.groupby(['Algorithm']).std().reset_index()['Northern_Abalone']
# sort and filter by 'Algorithm' and keys_list_boxplot
final_result_barchart = final_result_barchart[final_result_barchart['Algorithm'].isin(keys_list_boxplot)]
# sort by the order of keys_list_boxplot
final_result_barchart = final_result_barchart.set_index('Algorithm').reindex(keys_list_boxplot).reset_index()
# plot with matplotlib
ax = plt.figure(figsize=(10, 5))
final_result_barchart.plot(
    x='Algorithm', 
    y=['Sea_Otters', 'Northern_Abalone'], 
    kind='bar',  
    capsize=4, 
    rot=0, 
    yerr=final_result_barchart[['Sea_Otters_std', 'Northern_Abalone_std']].values.T,
    color=['darkred', 'orange'],
    ax = ax.gca()
    )
plt.legend(loc = 'upper right')
plt.show()

# benchmark: randomly
def test_benchmark_policy(fnPattern):
    random_df = read_csv(fnPattern)
    # Confidence interval for Sea_Otters and Northern_Abalone
    group_df = random_df.groupby(['step']).first().reset_index()[['step']]
    group_df['Sea_Otters_std'] = random_df.groupby(['step'])['Sea_Otters'].std().reset_index()['Sea_Otters']
    group_df['Northern_Abalone_std'] = random_df.groupby(['step'])['Northern_Abalone'].std().reset_index()['Northern_Abalone']
    group_df['Sea_Otters_mean'] = random_df.groupby(['step'])['Sea_Otters'].mean().reset_index()['Sea_Otters']
    group_df['Northern_Abalone_mean'] = random_df.groupby(['step'])['Northern_Abalone'].mean().reset_index()['Northern_Abalone']
    plt.figure(figsize=(10, 5))
    plt.plot(group_df['step'], group_df['Sea_Otters_mean'], label='Sea_Otters')
    plt.fill_between(group_df['step'], group_df['Sea_Otters_mean'] - group_df['Sea_Otters_std'], group_df['Sea_Otters_mean'] + group_df['Sea_Otters_std'], alpha=0.2)
    plt.plot(group_df['step'], group_df['Northern_Abalone_mean'], label='Northern_Abalone')
    plt.fill_between(group_df['step'], group_df['Northern_Abalone_mean'] - group_df['Northern_Abalone_std'], group_df['Northern_Abalone_mean'] + group_df['Northern_Abalone_std'], alpha=0.2)
    plt.legend(loc = 'best')
    plt.show()

test_benchmark_policy('reward_random')

# test: always take one action
# Action 0
test_benchmark_policy('reward_action0_run')

# Action 1
test_benchmark_policy('reward_action1_run')

# Action 2
test_benchmark_policy('reward_action2_run')

# Action 3
test_benchmark_policy('reward_action3_run')

# Action 4
test_benchmark_policy('reward_action4_run')
