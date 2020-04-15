import os.path
from os import path
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt
import pickle

data_1 = pd.read_csv('/Users/karanwadhwani/Documents/4-1/RL/AdaptiveDDPG/Reinforcement-learning-for-portfolio-allocation/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv')

equal_4711_list = list(data_1.tic.value_counts() == 4711)

names = data_1.tic.value_counts().index
select_stocks_list = list(names[equal_4711_list])+['NKE','KO']
data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]
data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]
data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']
all_data = data_3[(data_3.datadate > 20010000) & (data_3.datadate < 20190000)]
dates = all_data['datadate'].values.tolist()

tics = ['AXP', 'AAPL', 'VZ', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'DIS', 'DWDP', 'XOM', 'HD', 'INTC', 'IBM', 'JNJ', 'MCD', 'MRK', 'MMM', 'NKE', 'PFE', 'PG', 'UNH', 'UTX', 'WMT', 'WBA', 'MSFT', 'CSCO', 'GS']

sendf = pd.read_csv('sentiment_16.csv')

for i, row in sendf.iterrows():
    sentiments = []
    for tic in tics:
        csv_path = 'sentiment_'+tic+'.csv'
        #print(csv_path)
        if path.exists(csv_path):
            csv_df = pd.read_csv(csv_path)
            # print(csv_path)
            sentiments.append(csv_df.at[i, 'sentiment'])
        else:
            sentiments.append(0)
    sendf.at[i, 'sentiment'] = sentiments
    print(sendf.at[i, 'datadate'])
sendf.to_csv('sentiment_16.csv')
