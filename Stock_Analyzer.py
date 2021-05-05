# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:59:43 2021

@author: prab_
"""

import pandas as pd
import yfinance as yf
import numpy as np
import mplfinance as mpf

class Stock_Analyzer:
    def __init__(self,
                 tickers = ['TSLA'],
                 period = "6mo",
                 interval = "1d"):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.data = self.__read_data()
        self.open = self.data['Open']
        self.high = self.data['High']
        self.low = self.data['Low']
        self.close = self.data['Close']
    
    
    def __read_data(self):
        data = yf.download(tickers = self.tickers, period = self.period, interval = self.interval, group_by = "Ticker")
        return data

    
    def sma(self, period):
        #Use standard 50, 100, 200 for SMA
        sma = self.close.rolling(window=period).mean()
        self.data[f'SMA_{period}'] = sma
        return sma
    
    def ema(self, period):
        #Use standard 9, 12, 20 for EMA
        ema = self.close.ewm(span = period, adjust = False).mean()
        self.data[f'EMA_{period}'] = ema
        return ema
    
    def std_dev(self, period):
        std_dev = self.close.rolling(window=period).std()
        return std_dev
    
    def bollinger_band(self, period):
        #Use standard 20 period for Bollinger bands
        middle_band = self.sma(period)
        std_dev = self.std_dev(period)
        upper_band = middle_band + std_dev*2
        lower_band = middle_band - std_dev*2
        self.data[f'MIDDLE_B_{period}'] = middle_band
        self.data[f'UPPER_B_{period}'] = upper_band
        self.data[f'LOWER_B_{period}'] = lower_band
        return middle_band, upper_band, lower_band
    
    def atr(self, period):
        #Use standard 14 period for ATR
        ranges = pd.concat([self.high - self.low,
                         np.abs(self.high - self.close.shift()),
                         np.abs(self.low - self.close.shift())], axis = 1)
        true_range = np.max(ranges, axis = 1)
        #SMA ATR
        atr = true_range.rolling(period, min_periods = period).sum()/period
        #Part 2 SMA ATR for more accuracy, may not be correct
        #atr = (atr.shift()*(period - 1) + true_range)/period
        
        #RMA ATR
        #atr = true_range.ewm(alpha = 1/period, min_periods = period, adjust = False).mean()
        #Part 2 RMA ATR provides same results, so not needed
        #atr = (atr.shift()*(period - 1) + true_range)/period
        self.data[f'ATR_{period}'] = atr
        return atr
    
    
    
    def plot_data(self,extra_cols):
        apdict = mpf.make_addplot(self.data[extra_cols])
        mpf.plot(self.data, type = 'candlestick', volume = False, show_nontrading = False, addplot=apdict)
        
        #mpf.plot(self.data, type = 'candlestick', volume = False, show_nontrading = False)
        #apdict = mpf.make_addplot(self.data['Close'])
        #mpf.plot(self.data, volume = False, addplot = apdict, type = 'line', mav = (20))
        


test = Stock_Analyzer()
test.atr(20)
test.sma(20)
test.plot_data(['SMA_20', 'ATR_20'])



"""
test.ema(9)
test.ema(12)
test.plot_data(['EMA_9','EMA_12'])
print(test.data)
"""