#==============================================================================
"""Market Simulator based on a synthetic 'Market Order' data file

    1. Execution the orders specified in the Orders csv file
    2. Assessment of the portfolio value on a daily basis after 
        executing the orders
 
@author: Dhineshkumar"""
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

#==============================================================================
# FUNCTION DEFINITIONS: 
#==============================================================================

"""Function for returning CSV file path given a ticker symbol"""
def symbol_to_path(symbol, base_dir="data/"):
     return base_dir+"{}.csv".format(str(symbol))


"""Function for reading stock data (adjusted close) for given symbols. 
    Returns dataframe with only trading days"""
def get_data(syms, dates):
    symbols = syms.copy()
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['NaN'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
    
    #df = df.dropna() #Taking care of untraded days
    df['Cash'] = np.ones(len(df))
    return df

"""Function that returns a dataframe with # of shares traded and cash movement"""
def get_trades(syms, dates, df):
    
    symbols = syms.copy()
    trades = df.copy()
    trades[:] = 0.0
    
    order = pd.DataFrame(index=dates)
    order_temp = pd.read_csv(symbol_to_path('ORDERS'), index_col = 'Date',
                        parse_dates = True, na_values =['NaN'])
    order = order.join(order_temp)
    order = order.dropna()
   
    for dates in order.index:
        if order.loc[dates,'Order'] == 'SELL':
            trades.loc[dates,(order.loc[dates,'Symbol'])] = order.loc[dates,'Shares']*(-1.0)
            trades.loc[dates,'Cash'] = (order.loc[dates,'Shares'])*df.loc[dates,(order.loc[dates,'Symbol'])]
        elif order.loc[dates,'Order'] == 'BUY':
            trades.loc[dates,(order.loc[dates,'Symbol'])] = order.loc[dates,'Shares']
            trades.loc[dates,'Cash'] = order.loc[dates,'Shares']*df.loc[dates,(order.loc[dates,'Symbol'])]*(-1)
      
    return trades

"""Function that returns Portfolio Holdings"""    
def get_holdings(dates, trades):
    holdings = trades.copy()
    holdings.iloc[0]['Cash'] = 100000 + trades.iloc[0]['Cash']
    holdings = holdings.cumsum()
    holdings = holdings.dropna()
    return holdings

"""Function that returns Portfolio Values"""    
def get_values(prices, holdings):
    values = prices*holdings
    values = values.dropna()
    return values

"""Function for returning periodic returns - period specified"""
def compute_periodic_returns(df,sf=252.0):
    
    if sf == 252.0:
        returns = df.copy()
        returns = (returns/returns.shift(1))-1.0 # Daily Returns over previous trading day
        returns.iloc[0] = 0
    if sf == 52.0: 
        returns = df.copy()
        returns = returns.resample('W-FRI').sum() # Mean of values for each week Mon-Fri
        returns = (returns/returns.shift(1))-1.0 # Weekly Returns over previous week
        returns.iloc[0] = 0
    if sf == 12.0:
        returns = df.copy()
        returns = returns.resample('M').sum() # Mean of values for each month
        returns = (returns/returns.shift(1))-1.0 # Monthly returns over previous month
        returns.iloc[0] = 0
    return returns


"""Function for returning cumulative return values"""
def compute_cumulative_returns(df):
    cum_ret = df.copy()
    cumulative_return = cum_ret.iloc[-1]/cum_ret.iloc[0]-1.0
    return cumulative_return

"""Function for normalising data with respect to values on the first day / week / month """
def normalise(df):
    df = df/df.iloc[0]
    return df

"""Function for computing periodic sharpe ratio"""
def sharpe(port_val,rfr=0.0,sf=252.0):
    dailyrfr = ((1.0 + rfr)**(1.0/sf))-1.0
    ret = compute_periodic_returns(port_val,sf)
    numerator = (ret-dailyrfr).mean()
    denominator = ret.std()
    Annual_S = (sf**(1./2))*numerator/denominator # Annualized Sharpe Ratio
    return Annual_S

"""Function for plotting data"""
def plot_data(df, title="Stock Prices", ylabel = "Price", xlabel="Date"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.show()
#==============================================================================
# FUNCTION FOR PORTFOLIO ASSESSMENT:    
#==============================================================================
def port_eval(syms = ['AAPL','AXP', 'GOOG', 'GLD','HNZ','HPQ', 'IBM','XOM','SPY'],
                      sd=pd.to_datetime('2014.1.1'), ed=pd.to_datetime('2015.1.1'),
                      sv=1000000, rfr=0.0, sf=252,
                     gen_plot=True):
    symbols = syms.copy()
    dates = pd.date_range(sd,ed)

    prices = get_data(syms, dates) # GENERATING PRICES DATAFRAME 
     
    trades = get_trades(syms, dates, prices) # GENERATING TRADES DATAFRAME
    
    holdings = get_holdings(dates, trades) # GENERATING HOLDINGS DATAFRAME 
    
    values = get_values(prices, holdings) # GENERATING PORTFOLIO VALUES DATAFRAME

    port_val = values.sum(axis = 1)
    
    # CUMULATIVE RETURN
    cr = compute_cumulative_returns(port_val)
    
    # AVERAGE PERIOD RETURN
    apr = compute_periodic_returns(port_val,sf)
    mean_apr = apr.mean()
    sdev_apr = apr.std()
    
    port_alloc = port_val.to_frame(name = 'Portfolio')
    if gen_plot==True:
        spy = get_data(['SPY'],dates)
        df = port_alloc.join(spy)
        df = normalise(df)
        plot_data(df)
    
    # SHARPE RATIO
    sr = sharpe(port_val,rfr,sf)
    ev = port_val[-1] # END VALUE OF PORTFOLIO
   
    print("Start Date: "+str(sd)+"\n")
    print("End Date: "+str(ed)+"\n")
    print("Stocks Examined: "+str(symbols)+"\n")
    print("Sharpe Ratio: "+str(sr)+"\n")
    print("Volatility (stdev of periodic returns): "+str(sdev_apr)+"\n")
    print("Average Periodic Return: "+str(mean_apr)+"\n")
    print("Cumulative Return: " +str(cr)+"\n")
    print ("Starting Portfolio Value:"+ str(sv) +"\n")
    print ("Ending Portfolio Value:" + str(ev)+"\n")
   
    return cr,mean_apr,sdev_apr,sr,ev


portfolio = port_eval()

