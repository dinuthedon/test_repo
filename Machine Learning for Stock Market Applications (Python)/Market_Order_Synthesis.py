#==============================================================================
"""
Python Code for generating a sythetic Market Order Data

"""
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

df = pd.read_csv("data/AAPL.csv")

del df['Close'] 
del df['Volume'] 
del df['Adj Close']

df = df.rename(columns = {'Open':'Symbol', 'High':'Order', 'Low': 'Shares'})

item = ['AAPL','AXP', 'GOOG', 'GLD','HNZ','HPQ', 'IBM','XOM','SPY']*2000

df['Symbol'] = pd.Series(random.sample(item,len(df)))

item2 = ['BUY', 'SELL']*4000
df['Order'] = pd.Series(random.sample(item2,len(df)))

df['Shares'] = np.random.randint(90,150,len(df))

df = df.set_index('Date')

df.to_csv("data/ORDERS.csv")