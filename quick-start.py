#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
import collections
import json

import numpy as np
import pandas as pd
import pandasql as pdsql
import miceforest as mf
from warnings import simplefilter
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


# import all the useful subfunction in CDA
from utils.preprocess import *
from utils.baselines import *
from utils.utils import *

#------------------------------------------------------------------------------------------
#Experiment Setting Part: You can adjust these parameters directly to reset the experiments.
missing_rate = 0.3            # the ratio of missing values in raw table T.
pattern = 0                   # the pattern of missing values in raw table T. 0: MCAR, 1:MAR, 2: MNAR 
budget = 0.05                 # the ratio of missing values we can comlete by acquiring data
acquisition_rounds = 3        # the number of rounds to use up the budgets
dataset = 'tpcds'             # Here we have four datasets to select: 'census13', 'forest10', 'nursery', 'tpcds'
menu_type = 'RMenu'           # valid value including 'RMenu', 'OMenu', and 'SMenu'
confidence_interval = 0.9     # the confidence interval for conformal confidence control
#------------------------------------------------------------------------------------------

def main():
    # Before CDA, we prepare the dataset and menus. 
    ground_truth_data = load_dataset(dataset)                                                   # the ground truth of the dataset without any missing values
    incomplete_dataset =  generate_incomplete_data(ground_truth_data, missing_rate, pattern)    # generate the incomplete table T in the paper
    menu = generate_menu(ground_truth_data)                                                     # note that the menu is independent from the missing table
    queries = load_workload(dataset)                                                            # the workload including many queries, each query describe one data product

    # the options for the cda process including multiple possibilities
    '''
    If menu_type = 'Rmenu', we have the following baselines:
    (1) random_cda; (2) rgreedy; (3) pmab; (4) str_sample
    Please refer to the paper for the details of these methods. Besides, our method is (5) cmos
    '''
    st = time.time()
    acquired_df_miss = cmos(incomplete_dataset, ground_truth_data, budget, acquisition_rounds, queries, menu_type)
    # using baseline
    # acquired_df_miss = random_cda(incomplete_dataset, ground_truth_data, budget, acquisition_rounds, queries, menu_type)
    et = time.time()

    pres_original, rs_original = get_acc(queries, incomplete_dataset, ground_truth_data)
    pres, rs = get_acc(queries, acquired_df_miss, ground_truth_data)  

    print("Without CDA, the recall is "+ str(pres_original))
    print("With CDA, the recall is "+ str(pres))
    print("The running time is" + str(et-st))


if __name__ == '__main__':
    main() 





