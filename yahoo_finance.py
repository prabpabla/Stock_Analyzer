# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:59:49 2021

@author: prab_
"""

"""
Get a list of tickers
Request data for a select timeframe, start date, end date, period, interval, open, close
Output 
Output a chart of the tickers
Return a list of metrics  
"""

import pandas as pd
import pprint
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import mplfinance as mpf
from enum import Enum


class Stock_Analyzer_Metric(Enum):
    EMA_9 = 1
    EMA_12 = 2
    SMA_50 = 3
    SMA_100 = 4
    SMA_200 = 5
    PERCENT_CHANGE_DAILY = 11
    PERCENT_CHANGE_WEEK = 12
    PERCENT_CHANGE_MONTH = 13
    PERCENT_CHANGE_6MONTH = 14

class Stock_Analyzer:
    def __init__(self,
                 tickers = ['TSLA'],
                 period = "6mo",
                 interval = "1d",
                 metrics=[Stock_Analyzer_Metric(1)]):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.metrics = metrics
    
    def read_data(self):
        data = yf.download(tickers = self.tickers, period = self.period, interval = self.interval, group_by = "Ticker")
        return data
    
    def close_data(self):
        close_data = []
        close_data = self.read_data().get("Close")
        return close_data
    
    def date(self):
        date = self.read_data()
        date2 = date.index
        return date2
    
    def sma(self):
        data = self.close_data() 
        sma = []
        sma_period = 20
        count = 0
        for i in range(data.size):
            if data[i] is None:
                sma.append(None)
            else:
                count += 1
                if count < sma_period:
                    sma.append(None)
                else:
                    sma.append(np.mean(data[i-sma_period+1:i+1]))
        return sma
    
    def sma_2(self):
        data = self.close_data()
        sma = []
        sma_period = 20
        sma = data.rolling(window=sma_period).mean()
        #sma.head(sma_period)
        return sma
            
    def data_to_csv(self):
        data = []
        for ticker in self.tickers:
            data = yf.download(tickers = ticker, period = self.period, interval = self.interval, group_by = "Ticker")
            #data = data[['Open','Close','High','Low']]
            #print(data)
            data.to_csv(f'ticker_{ticker}.csv')
            
    def plot_sma(self):
        date = self.date()
        my_year_month_fmt = mdates.DateFormatter('%m/%y')
        fig, sma_plot = plt.subplots(figsize=(16,9))
        data = self.close_data()
        sma2 = self.sma_2()
        sma1 = self.sma()
        #for i in range(1,len(data)):
            #if data[i] is None:
                #data[i] = data[i-1]
        
        xlabel = []
        xticks = [];
        
        for i in range(len(date)):
            xticks.append(i)
            xlabel.append(str(date[i]))
        
        test_sma = []
        for i in range(len(date)):
            test_sma.append(sma1[i])
            
        test_data = []
        for i in range(len(date)):
            test_data.append(data[i])
        sma_plot.set_xticks(xticks)
        #sma_plot.set_xlabel(xlabel)
        sma_plot.plot(xticks, test_data, label = 'Price', marker='.', linestyle='-')
        sma_plot.plot(xticks, test_sma, label = '50-days SMA', marker='.', linestyle='')
        sma_plot.plot(xticks, sma2, label = '50-days SMA')
        #sma_plot.xaxis.set_major_formatter(my_year_month_fmt)
        
        #mpf.plot(test_data)
        
    #def data_to_cvs(self):
test = Stock_Analyzer()
#print(test.read_data())
print(test.date())
#print(test.close_data())
print(test.sma_2())
#print(test.sma())
test.plot_sma()