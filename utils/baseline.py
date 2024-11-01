#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import torch
import csv
import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from collections import Counter
#from utils.data_loaders import dataset_loader

from utils.utils import *
from utils.mab import UCBMultiArmBandit

## Setting up and initialization

# Fix the seed ------------------------------------------------------
np.random.seed(42)
multi_imp_num = 1
multi_imp_iter = 1

class UCBMultiArmBandit:
    def __init__(self, num_arms, init_rewards):
        self.num_arms = num_arms
        self.total_rewards = np.zeros(num_arms)
        for i in range(len(init_rewards)):
            self.total_rewards[i] = init_rewards[i]
        self.arm_counts = np.zeros(num_arms)
        self.total_plays = 0

    def select_arm(self):
        if 0 in self.arm_counts:
            # Play each arm once to initialize
            return np.argmin(self.arm_counts)
        else:
            ucb_values = self.total_rewards / self.arm_counts + np.sqrt(2 * np.log(self.total_plays) / self.arm_counts)
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_rewards[chosen_arm] += reward
        self.arm_counts[chosen_arm] += 1
        self.total_plays += 1



def random_value_acquisition(df_miss, df_arr, budget, queries, W):
    miss_rows, miss_cols = np.where(np.isnan(df_miss.to_numpy()))
    budget_unit = int(len(miss_rows)*budget)
    acq_indexes = random.sample(range(0, len(miss_rows)), budget_unit)
    acquired_df_miss = df_miss.copy()

    for i in acq_indexes:
        acquired_df_miss.iat[miss_rows[i], miss_cols[i]] = df_arr.iat[miss_rows[i], miss_cols[i]]
        
    return acquired_df_miss


def random_sample_acquisition(df_miss, df_arr, budget, query):
    df_miss_copy = df_miss.copy()
    rids = random.sample(range(len(df_miss)), int(len(df_miss)*budget))
    df_miss_copy.iloc[rids] = df_arr.iloc[rids]
    
    return df_miss_copy
    
def greedy_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    columns = []
    for query in queries:
        query_columns = [q['col'] for q in query]
        columns += query_columns

    total_acq_num = budget*df_miss.isna().sum().sum()      # get acquisition cell number
    columns_count = dict(Counter(columns))
    for k, v in columns_count.items():
        _rows = np.where(np.isnan(df_miss_copy[k].to_numpy())) # get rows for missing column k
        sample_num = int(v / len(columns)*total_acq_num)       # get acquisition cell number in column k
        sample_rows = random.sample(range(len(_rows[0])), sample_num)
        for r in sample_rows:
            df_miss_copy[k].iat[_rows[0][r]] = df_arr[k].iat[_rows[0][r]]
    
    return df_miss_copy


def mab_acquisition(df_miss, df_arr, budget, queries, W):
    df_miss_copy = df_miss.copy()
    columns = []
    for query in queries:
        query_columns = [q['col'] for q in query]
        columns += query_columns
    columns_count = dict(Counter(columns))
    initial_rewards = list(columns_count.values())
    arm_labels = list(columns_count.keys())

    # initial UCB
    num_arms = len(arm_labels)
    bandit = UCBMultiArmBandit(num_arms, initial_rewards)

    # Simulate bandit plays
    batch_size_per_iter = 20 # change it according to your need
    num_plays = int(budget*df_miss.isna().sum().sum() / batch_size_per_iter)

    for _ in range(num_plays):
        chosen_arm = bandit.select_arm()

        # the reward should be if it has found a usable tuple
        reward = 0
        col = arm_labels[chosen_arm]
        missing_index = np.where(df_miss_copy[col].isna())[0]
        acquisition_index = np.random.choice(missing_index, size=batch_size_per_iter, replace=False)
        for ind in acquisition_index:
            df_miss_copy[col].iat[ind] = df_arr[col].iat[ind]
        imputed_part = df_miss_copy.loc[acquisition_index]
        for i, query in enumerate(queries):
            # get the dataframe of acquired answer and ground truth
            data_product = query_on_df(query, imputed_part)
            reward += len(data_product)*W[i]

        bandit.update(chosen_arm, reward)
    
    return df_miss_copy