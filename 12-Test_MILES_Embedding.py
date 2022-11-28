# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:58:59 2021

@author: Ward
"""

import time
import numpy as np
import pandas as pd
start_time = time.time()
pd.set_option('display.max_columns', None)

# Load np array
features = np.load('../10_data/features.npy', allow_pickle=True)

bags_train = np.load('../10_data/bags_train.npy', allow_pickle=True)
id_bags_train = np.load('../10_data/id_bags_train.npy', allow_pickle=True)
y_bag_train = np.load('../10_data/y_bag_train.npy', allow_pickle=True).astype('int')
y_ins_train = np.load('../10_data/y_ins_train.npy', allow_pickle=True)
bags_validation = np.load('../10_data/bags_validation.npy', allow_pickle=True)
id_bags_validation = np.load('../10_data/id_bags_validation.npy', allow_pickle=True)
y_bag_validation = np.load('../10_data/y_bag_validation.npy', allow_pickle=True).astype('int')
y_ins_validation = np.load('../10_data/y_ins_validation.npy', allow_pickle=True)

training_set_size = 1000
validation_set_size = 300

if training_set_size:
    bags_train = bags_train[:training_set_size]
    id_bags_train = id_bags_train[:training_set_size]
    y_bag_train = y_bag_train[:training_set_size]
    y_ins_train = y_ins_train[:training_set_size]
    bags_validation = bags_validation[:validation_set_size]
    id_bags_validation = id_bags_validation[:validation_set_size]
    y_bag_validation = y_bag_validation[:validation_set_size]
    y_ins_validation = y_ins_validation[:validation_set_size]


## STANDARDIZE

from preprocessing.standarize_bags import StandarizerBagsList

bag_standardizer = StandarizerBagsList()

bag_standardizer.fit(bags_train)

bags_train_std = bag_standardizer.transform(bags_train)
bags_validation_std = bag_standardizer.transform(bags_validation)


## MILES EMBEDDING

from bag_representation.miles_mapping import MILESMapping

bag_miles_mapping = MILESMapping(sigma2=4.5**2)

bag_miles_mapping.fit(bags_train_std)

bags_train_std_miles, bags_train_std_miles_closest = bag_miles_mapping.transform(bags_train_std)

bags_validation_std_miles, bags_validation_std_miles_closest = bag_miles_mapping.transform(bags_validation_std)








