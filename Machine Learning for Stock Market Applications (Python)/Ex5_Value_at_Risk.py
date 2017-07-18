#==============================================================================
""" 1-Week 90% Value at Risk calculation of a Portfolio

    Step-1: Optimizing portfolio weights based on Time Series Data 
    Stet-2: Converting daily stock prices to weekly prices
    Step-3: Data of the latest week is fixed as that of Time-0 
    Step-4: Finding covariance matrix of the protfolio stock prices (mean-scaled)
            (Decay weight used for weekly data = 90%)
    Step-5: Finding standard deviation of the portfolio value 
            (Time-1, the next week)
    Step-6: Finding 0.1 Quantile of the portfolio value (Time-1, the next week)
    Step-7: 1-Week 90% VaR = 
                (Portfolio value at Time-0 - 0.1% Quantile at Time-1)
 
@author: Dhinesh"""
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy.optimize import minimize


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
    
    df = df.dropna() #Taking care of untraded days
    return df

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

"""Function for normalising data with respect to mean"""
def mean_normalize(df):
    df_mean = df.mean()
    df = df - df.mean()
    return df_mean, df

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
# PORTFOLIO ASSESSMENT:    
#==============================================================================

"""Function that returns general assessment of predefined portfolio:
    1. Cumulative return
    2. Average daily return
    3. Standard deviation of daily returns
    4. Annualized Sharpe Ratio
    5. Final portfolio value """
    
def assess_portfolio(sd, ed,
                     syms,
                     allocs,
                     sv, rfr, sf,
                     gen_plot=False): # All these values are from problem definition
    
    # GET DATA  
    symbols = syms.copy()
    dates = pd.date_range(sd,ed)
    df = get_data(syms,dates)
    if 'SPY' not in symbols: # DELETE SPY AS WE USE IT AS A BENCHMARK
        del df['SPY']
    
    # DATA CLEANING
    if df.isnull().values.any() == True : 
        df.fillna(method="ffill", inplace="True")
        df.fillna(method="bfill", inplace="True") # In case data is missing at the beginning

    # DATA NORMALISATION
    df = normalise(df)

    # PORTFOLIO VALUE CALCULATION
    allocation = allocs*df
    pos_vals = allocation*sv # Captures cash movement over assessment period
    port_val = pos_vals.sum(axis = 1) # Daily / Weekly / Monthly  portfolio value 
    
    # CUMULATIVE RETURN
    cr = compute_cumulative_returns(port_val)
    
    # AVERAGE PERIOD RETURN
    apr = compute_periodic_returns(port_val,sf)
    mean_apr = apr.mean()
    sdev_apr = apr.std()
    
    # NORMALISED PORTFOLIO
    port_alloc = allocation.sum(axis=1)
    port_alloc = port_alloc.to_frame(name = 'Portfolio')
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
    print("Allocations: "+str(allocs)+"\n")
    print("Sharpe Ratio: "+str(sr)+"\n")
    print("Volatility (stdev of daily returns): "+str(sdev_apr)+"\n")
    print("Average daily Return: "+str(mean_apr)+"\n")
    print("Cumulative Return: " +str(cr)+"\n")
    print ("Starting Portfolio Value:"+ str(sv) +"\n")
    print ("Ending Portfolio Value:" + str(ev)+"\n")
    
    return cr,mean_apr,sdev_apr,sr,ev

#==============================================================================
# OPTIMISING FOR PORTFOLIO ALLOCATIONS:    
#==============================================================================

"""Function that returns Negative Sharpe Ratio for normalised portfolio"""
def sharpe_for_optimisation(allocations, normalised_df):
    alloc = allocations*normalised_df
    port_val = alloc.sum(axis = 1) # Portfolio value based on defined weights
    return -1*sharpe(port_val)
   
""" Function that Optimizes Portfolio"""    
def optimize_portfolio(sd, ed,
                       syms, gen_plot=False):
    # GET DATA  
    symbols = syms.copy()
    dates = pd.date_range(sd,ed)
    df = get_data(syms,dates)
    if 'SPY' not in symbols: # DELETE SPY AS WE USE IT AS A BENCHMARK
        del df['SPY']
    
    # DATA CLEANING
    if df.isnull().values.any() == True : 
        df.fillna(method="ffill", inplace="True")
        df.fillna(method="bfill", inplace="True") # In case data is missing at the beginning
        
    # DATA NORMALISATION
    df = normalise(df)
    #print(df.tail())
    
    # INITIAL PORTFOLIO WEIGHTS 
    allocs_init = np.array([(1./len(symbols))]*len(symbols)) # EVEN PORTFOLIO WEIGHTS
    
    # RANGES FOR PORTFOLIO WEIGHTS
    bounds = ((0.,1.),)*len(symbols) # Upper and Lower bounds on weights
    
    # OPTIMIZATION OF PORTFOLIO WEIGHTS: MINIMISATIION OF NEGATIVE SHARPE RATIO
    allocs = minimize(sharpe_for_optimisation, allocs_init, args = (df,),
                      method='SLSQP', options = {'disp':False}, bounds = bounds,
                      constraints = ({ 'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs) }))
                        # Constraint that sum of weights = 1.0
    allocs = allocs.x
    
    
    # ASSESSMENT OF OPTIMIZED PROTFOLIO
    cr,apr,sdev_apr,sr,ev = assess_portfolio(sd,ed,symbols,allocs,1000000,0.0,252.0,gen_plot = True)       
    
    return allocs,cr,apr,sdev_apr,sr,ev

#==============================================================================
# VALUE AT RISK CALCULATION:    
#==============================================================================

""" Function for VaR calculation"""  

def VaR_calc(sd, ed, syms, gen_plot = False):
    
    allocs,_,_,_,_,ev = optimize_portfolio(sd, ed, syms)
    
    # GET DATA  
    symbols = syms.copy()
    dates = pd.date_range(sd,ed)
    df = get_data(syms,dates)
    if 'SPY' not in symbols: # DELETE SPY AS WE USE IT AS A BENCHMARK
        del df['SPY']
    
    # DATA CLEANING
    if df.isnull().values.any() == True : 
        df.fillna(method="ffill", inplace="True")
        df.fillna(method="bfill", inplace="True") # In case data is missing at the beginning
    df = df.resample('W').mean() # Weekly average sammpling
  
    #Data Mean Normalaization
    df_mean, df = mean_normalize(df)
    
    # Generating decay weights for weeks
    T,n = df.shape
    w = np.ones((T,1))
    
    for i in range(1,T):
        w[i] = w[i-1]*0.95
    
    w = np.flipud(w) # Latest week has 100% weight while previous
                            # weeks have 95% weight over subequent weeks
    
    # Covariance Matrix at time T = 52 i.e. latest week
    sum_w = w.sum()
    sigma = np.zeros((n,n))
        
    for i in range(0,n):
        for j in range(0,n):
            
            sigma[i,j] = (w[:,0]*(df.iloc[:,i])*(df.iloc[:,j])).sum()/sum_w
            #w = (T,1), df.iloc[:,i] = (T,1), df.iloc[:,j] = (T,1)
            
    #print("The Covariance Matrix is: \n {}".format(sigma))
    
    # Portfolio Value
    pos_vals = allocs*ev # Captures cash distribution among portfolio
    port_val = pos_vals.sum() # Portfolio value at latest week

    # Standard Deviation of the portfolio  
    pos_vals = pos_vals.reshape(n,1)
    sdev_sq = np.matmul((np.matmul(pos_vals.T,sigma)),pos_vals)
    sdev = sdev_sq**0.25

    print("Standrd Deviation of Portfolio value (1-week outlook): $ {} \n"
          .format(sdev.item(0)))
    
    # 0.10 Quantile of the portfolio
    quantile = ev - sdev*1.282
    
    print("The 0.10 quantile of Portfolio value (1-week outlook): $ {} \n"
          .format(quantile.item(0)))
    
    # 1-week 90% USD VaR of the Portfolio
    VaR = ev - quantile
  
    print("1-week 90% USD VaR of Portfolio: $ {} \n".format(VaR.item(0)))
    
    
VaR_calc(sd=pd.to_datetime('2010.1.1'), ed=pd.to_datetime('2010.12.31'),
                       syms=['GOOG','AAPL','GLD','XOM'])
    
    