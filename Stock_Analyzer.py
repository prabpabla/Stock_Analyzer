# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:59:43 2021

@author: prab_
"""

import pandas as pd
import yfinance as yf
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import math

class Stock_Analyzer:
    def __init__(self,
                 tickers = ['NIO'],
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
        self.volume = self.data['Volume']
    
    def __read_data(self):
        data = yf.download(tickers = self.tickers, period = self.period, interval = self.interval, group_by = "Ticker", prepost = True)
        return data

    def sma(self, period):
        #SIMPLE MOVING AVERAGE
        #Use standard 50, 100, 200 periods for SMA
        sma = self.close.rolling(window = period).mean()
        self.data[f'SMA_{period}'] = sma
        return sma
    
    def ema(self, period):
        #EXPONENTIAL MOVING AVERAGE
        #Use standard 9, 12, 20 periods for EMA 
        ema = self.close.ewm(span = period, adjust = False, min_periods = period).mean()
        #ema = (self.close*multiplier) + (ema[-1]*(1-multiplier))
        """
        #EMA CALC 2
        multiplier = (2/(period + 1))
        ema = [sum(self.close[:period])/period]
        for i in self.close[1:]:
            ema.append((i*multiplier) + ema[-1]*(1 - multiplier))
        """
        """
        #EMA CALC 3
        multiplier = (2/(period + 1))
        ema_cur = self.sma(period)
        for i in self.close[-period:]:
            ema = ((i*multiplier) + ema_cur*(1 - multiplier))
        """
        self.data[f'EMA_{period}'] = ema
        return ema
    
    def dema(self, period):
        #DOUBLE EXPONENTIAL MOVING AVERAGE
        ema_of_ema = self.ema(period).ewm(span = period, adjust = False).mean()
        dema = 2*self.ema(period) - ema_of_ema
        self.data[f'DEMA_{period}'] = dema
    
    def tema(self, period):
        #TRIPLE EXPONENTIAL MOVING AVERAGE
        ema_of_ema = self.ema(period).ewm(span = period, adjust = False).mean()
        ema_of_ema_of_ema = ema_of_ema.ewm(span = period, adjust = False).mean()
        tema = 3*self.ema(period) - 3*ema_of_ema + ema_of_ema_of_ema
        self.data[f'TEMA_{period}'] = tema
    
    def std_dev(self, period):
        #STANDARD DEVIATION
        std_dev = self.close.rolling(window = period).std()
        return std_dev
    
    def b_band(self, period):
        #BOLLINGER BAND
        #Use standard 20 period for Bollinger bands
        middle_band = self.sma(period)
        std_dev = self.std_dev(period)
        upper_band = middle_band + std_dev*2
        lower_band = middle_band - std_dev*2
        self.data[f'MIDDLE_BB_{period}'] = middle_band
        self.data[f'UPPER_BB_{period}'] = upper_band
        self.data[f'LOWER_BB_{period}'] = lower_band
        return middle_band, upper_band, lower_band
    
    def b_bandwidth(self, period):
        #BOLLINGER BANDWIDTH
        middle_band, upper_band, lower_band = self.b_band(period)
        bbw = ((upper_band - lower_band)/middle_band)*100
        self.data[f'B_BW_{period}'] = bbw
        return bbw
    
    def percent_bb(self, period):
        #PERCENT BOLLINGER BAND
        middle_band, upper_band, lower_band = self.b_band(period)
        pbb = (self.close - lower_band)/(upper_band - lower_band)
        self.data[f'%_BB_{period}'] = pbb
        return pbb
    
    def atr(self, period):
        #AVERAGE TRUE RANGE
        #Use standard 14 period for ATR
        ranges = pd.concat([self.high - self.low,
                         np.abs(self.high - self.close.shift()),
                         np.abs(self.low - self.close.shift())], axis = 1)
        true_range = np.max(ranges, axis = 1)
        
        #SMA ATR CALC
        atr = true_range.rolling(period, min_periods = period).sum()/period
        #Part 2 SMA ATR for more accuracy, may not be correct
        #atr = (atr.shift()*(period - 1) + true_range)/period
        
        #RMA ATR CALC
        #atr = true_range.ewm(alpha = 1/period, min_periods = period, adjust = False).mean()
        #Part 2 RMA ATR provides same results, so not needed
        #atr = (atr.shift()*(period - 1) + true_range)/period
        self.data[f'ATR_{period}'] = atr
        return atr
    
    def wma(self, period):
        #WEIGHTED MOVING AVERAGE
        #Use standard 20, 50, 100, 200 periods for WMA
        weights = np.arange(1, period +1)
        wma = self.close.rolling(window = period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw = True)
        self.data[f'WMA_{period}'] = wma
        return wma
    
    def hma(self, period):
        #HULL AVERAGE AVERAGE
        #Use standard 20, 50, 100, 200 periods for HMA
        wma_1 = self.wma(round(period/2))
        wma_2 = self.wma(period)
        raw_hma = wma_1*2 - wma_2
        
        #Part 2 WMA(sqrt(period)) of RAW HMA
        sqrt_period = round(math.sqrt(period))
        weights = np.arange(1, sqrt_period +1)
        hma = raw_hma.rolling(window = sqrt_period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw = True)
        self.data[f'HMA_{period}'] = hma
        return hma
    
    def keltner(self, period_E, period_A):
        #KELTNER CHANNEL
        #Use standard 20 period for EMA, and 10 period for ATR
        middle_channel = self.ema(period_E)
        upper_channel = middle_channel + self.atr(period_A)*2
        lower_channel = middle_channel - self.atr(period_A)*2
        self.data[f'MIDDLE_K_{period_E}'] = middle_channel
        self.data[f'UPPER_K_{period_E}'] = upper_channel
        self.data[f'LOWER_K_{period_E}'] = lower_channel
        return middle_channel, upper_channel, lower_channel
    
    def cci(self, period):
        #COMMODITY CHANNEL INDEX
        #Use standard 20 period for CCI
        tp = (self.high + self.low + self.close)/3
        #tp = self.close
        const = 0.015
        #mean_dev = tp.rolling(window = period).apply(lambda x: pd.Series(x).mad())
        #cci = (tp - self.sma(period)/(const * mean_dev))/period
        cci = pd.Series((tp - tp.rolling(window = period).mean())/(const*tp.rolling(window = period).std()))
        #Limit CCI between 100 and -100
        for i in range(len(cci)):
            if cci[i] >= 100: cci[i] = +100
            if cci[i] <= -100: cci[i] = -100
        self.data[f'CCI_{period}'] = cci
        return cci
        
    def mae(self, period, percent):
        #MOVING AVERAGE ENVELOPE
        #Use standard 20 for SMA period, and 2.5% envelope
        middle_envelope = self.sma(period)
        upper_envelope = middle_envelope*(1 + percent/100)
        lower_envelope = middle_envelope*(1 - percent/100)
        self.data[f'MIDDLE_E_{period}'] = middle_envelope
        self.data[f'UPPER_E_{period}'] = upper_envelope
        self.data[f'LOWER_E_{period}'] = lower_envelope
        return middle_envelope, upper_envelope, lower_envelope
    
    def emae(self, period, percent):
        #EXPONENTIAL MOVING AVERAGE ENVELOPE
        #Use standard 20 for period, and 2.5% envelope
        middle_envelope = self.ema(period)
        upper_envelope = middle_envelope*(1 + percent/100)
        lower_envelope = middle_envelope*(1 - percent/100)
        self.data[f'MIDDLE_EE_{period}'] = middle_envelope
        self.data[f'UPPER_EE_{period}'] = upper_envelope
        self.data[f'LOWER_EE_{period}'] = lower_envelope
        return middle_envelope, upper_envelope, lower_envelope
    
    def ppo(self):
        #PERCENTAGE PRICE OSCILLATOR
        #Use standard 12, 26 for EMA period, 9 for Signal Line EMA period
        ppo = (self.ema(12) - self.ema(26))/(self.ema(26))*100
        signal_line = ppo.ewm(span = 9, adjust = False, min_periods = 9).mean()
        #ppo_hist = ppo - signal_line
        self.data['PPO'] = ppo
        self.data['PPO_SIG'] = signal_line
        return ppo, signal_line
    
    def macd(self):
        #MOVING AVERAGE CONVERGENCE/ DIVERGENCE
        #Use standard 12, 26 for EMA period, 9 for Signal Line EMA period
        macd = self.ema(12) - self.ema(26)
        signal_line = macd.ewm(span = 9, adjust = False, min_periods = 9).mean()
        #macd_hist = macd - signal_line
        self.data['MACD'] = macd
        self.data['MACD_SIG'] = signal_line
        return macd, signal_line
    
    def price_c(self, period):
        #PRICE DONCHIAN CHANNEL
        #Use standard 20, 10 for period
        upper_channel = self.high.rolling(window = period).max()
        lower_channel = self.low.rolling(window = period).min()
        middle_channel = (upper_channel + lower_channel)/2
        self.data[f'MIDDLE_DC_{period}'] = middle_channel
        self.data[f'UPPER_DC_{period}'] = upper_channel
        self.data[f'LOWER_DC_{period}'] = lower_channel
        return middle_channel, upper_channel, lower_channel
    
    def kama(self, period_ER, period_F, period_S):
        #KAUFMAN'S ADAPTIVE MOVING AVERAGE
        #Use standard 10 period for Efficiency Ratio, 2, 30 for fast and slow EMA, respectively
        change = abs(self.close - self.close.shift(period_ER))
        volatility = (abs(self.close - self.close.shift())).rolling(period_ER).sum()
        eff_ratio = change/volatility
        fast, slow = 2/(period_F + 1), 2/(period_S + 1)
        smooth_const = (eff_ratio*(fast - slow) + slow)**2
        kama = self.sma(period_ER)
        kama = kama.shift() + smooth_const*(self.close - kama.shift())
        self.data['KAMA'] = kama
        return kama
        
    def aroon(self, period):
        #AROON
        #Use standard 25 days for length
        day_high = self.high.rolling(period).max()
        for i in range(period, -1, -1):
            day_high.loc[self.high.shift(i) == day_high] = i
        day_high = (period - day_high)*4
        self.data['AROON_UP'] = day_high
        
        day_low = self.low.rolling(period).min()
        for i in range(period, -1, -1):
            day_low.loc[self.low.shift(i) == day_low] = i
        day_low = (period - day_low)*4
        self.data['AROON_DOWN'] = day_low
        return day_high 
    
    def vol_by_price(self, p_range):
        #VOLUME BY PRICE
        #Plots a horizontal histogram, showing the amount of activity at price intervals
        lowest_low = round(self.low.min())
        highest_high = round(self.high.max())
        range_div = (highest_high - lowest_low)/p_range
        
        price_range = []
        for i in range(p_range):
            price_range.append(lowest_low + range_div*i)
        #print(price_range)
        
        volume_range = [0]*p_range
        for i in range(len(price_range)):
            for j in range(len(self.close)):
                #print(self.close[j] , ',' , price_range[i])
                if (price_range[i] <= self.close[j] <= (price_range[i] + range_div)):
                    volume_range[i] = volume_range[i] + self.volume[j]
        #print(volume_range)
        plt.barh(price_range, volume_range)
        
    def vwap(self):
        #VOLUME WEIGHTED AVERAGE PRICE
        #Based on total dollar value of all trades for current day, divided by total trading volume for current day
        typical_price = (self.high + self.low + self.close)/3
        """
        vol_price = typical_price * self.volume
        total_vol_price = vol_price.copy()
        for i in range(len(typical_price)):
            if i > 0:
                total_vol_price[i] = total_vol_price[i] + total_vol_price[i-1]
        
        total_vol = self.volume.copy()
        for i in range(len(typical_price)):
            if i > 0:
                total_vol[i] = total_vol[i] + total_vol[i-1]
                
        vwap = total_vol_price/total_vol
        print(vol_price, typical_price, self.volume)
        """
        vwap = (np.cumsum(self.volume*typical_price)/(np.cumsum(self.volume))).ffill()
        #vwap = (self.volume*typical_price).rolling(20).sum()/self.volume.rolling(20).sum()
        self.data['VWAP'] = vwap
        return vwap
     
    def adl(self):
        #ACCUMULATION DISTRIBUTION LINE
        #Volume based indicator designed to measure the cumulative flow of money 
        #Money Flow Multiplier, Money Flow Volume
        mfm = ((self.close - self.low) - (self.high - self.close))/(self.high - self.low)
        mfv = mfm*self.volume
        adl = mfv.copy()*0
        adl = adl.shift() + mfv
        self.data['ADL'] = adl
        return adl
    
    def obv(self):
        #ON BALANCE VOLUME
        #Measures buying and selling pressure as a cumulative indicator
        obv = self.volume.copy()
        for i in range(len(obv)):
            if i > 0:
                if self.close[i] > self.close[i-1]:
                    obv[i] = obv[i-1] + self.volume[i]
                if self.close[i] < self.close[i-1]:
                    obv[i] = obv[i-1] - self.volume[i]    
        self.data['OBV'] = obv
        return obv
    
    def cmf(self):
        #CHAIKIN MONEY FLOW
        #Measures the amount of Money Flow Volume over a specific period
        #Money Flow Multiplier, Money Flow Volume
        mfm = ((self.close - self.low) - (self.high - self.close))/(self.high - self.low)
        mfv = mfm*self.volume
        cmf = mfv.rolling(20).sum()/self.volume.rolling(20).sum()
        self.data['CMF'] = cmf
        return cmf
    
    def chaikin(self):
        #CHAIKIN OSCILLATOR
        #Measures the momentum of the Accumulation Distribution Line using the MACD formula
        adl_3_ema = self.adl().ewm(span = 3, adjust = False, min_periods = 3).mean()
        adl_10_ema = self.adl().ewm(span = 10, adjust = False, min_periods = 10).mean()
        cosc = adl_3_ema - adl_10_ema
        self.data['COSC'] = cosc
        return cosc
    
    def bop(self):
        #BALANCE OF POWER, or BALANCE OF MARKET POWER (BMP)
        #Oscillator that measures the strength of buying and selling pressure
        bop = ((self.close - self.open)/(self.high - self.low))
        self.data['BOP'] = bop
        return bop
    
    def pvo(self):
        #PERCENT VOLUME OSCILLATOR
        #Use standard 12, 26 for EMA period, 9 for Signal Line EMA period
        vol_ema_12 = self.volume.ewm(span = 12, adjust = False, min_periods = 12).mean()
        vol_ema_26 = self.volume.ewm(span = 26, adjust = False, min_periods = 26).mean()
        pvo = (vol_ema_12 - vol_ema_26)/(vol_ema_26)*100
        signal_line = pvo.ewm(span = 9, adjust = False, min_periods = 9).mean()
        #ppo_hist = pvo - signal_line
        self.data['PVO'] = pvo
        self.data['PVO_SIG'] = signal_line
        return pvo, signal_line
    
    def roc(self, period):
        #RATE OF CHANGE
        #Pure momentum oscillator, measures percent change in price from one period to next
        roc = (self.close - self.close.shift(period))/(self.close.shift(period))*100
        """
        roc = self.close.copy()*0
        for i in range(len(roc)):
            if i > period:
                roc[i] = (self.close[i] - self.close[i - period])/(self.close[i - period])*100
        """
        self.data['ROC'] = roc
        return roc
    
    def kst(self):
        #KNOW SURE THING, or SUMMED RATE OF CHANGE
        #Weighted average of four different rate-of-change values that have been smoothed
        #Use standard 10, 15, 20, 30 for ROC period, 10, 10, 10, 15 for SMA period, 9 for Signal Line
        rcma1 = self.roc(10).rolling(10).mean()
        rcma2 = self.roc(15).rolling(10).mean()
        rcma3 = self.roc(20).rolling(10).mean()
        rcma4 = self.roc(30).rolling(15).mean()
        kst = rcma1 + rcma2*2 + rcma3*3 + rcma4*4
        signal_line = kst.rolling(9).mean()
        self.data['KST'] = kst
        self.data['KST_SIG'] = signal_line
        return kst, signal_line
    
    def pivot_point(self):
        #PIVOT POINTS
        #Provides pivot points per month
        p_points = (self.high + self.low + self.close)/3
        high = self.high + 0
        low = self.low + 0
        
        date = p_points.index;
        pivotIndex = -1
        activeMonth = -1
        
        for i in range(len(p_points)-1,-1,-1):
            tickMonth = date[i].month
            if(activeMonth!=tickMonth):
                activeMonth = tickMonth
                pivotIndex = i
                j = i               
                while(j>0):
                    previousTickMonth = date[j].month;
                    if(previousTickMonth!=activeMonth):
                        break
                    pivotIndex = j
                    j = j - 1
            
            p_points[i] =  p_points[pivotIndex]
            high[i] = high[pivotIndex]
            low[i] = low[pivotIndex]

        s1 = p_points*2 - high
        r1 = p_points*2 - low
        s2 = p_points - (high - low)
        r2 = p_points + (high - low)
                  
        self.data['P_P'] = p_points
        self.data['S1'] = s1
        self.data['R1'] = r1
        self.data['S2'] = s2
        self.data['R2'] = r2
        return p_points
    
    def plot_data(self,extra_cols):
        apdict = mpf.make_addplot(self.data[extra_cols])
        mpf.plot(self.data, type = 'candlestick', volume = False, show_nontrading = False, addplot=apdict)
        
        #mpf.plot(self.data, type = 'candlestick', volume = False, show_nontrading = False)
        #apdict = mpf.make_addplot(self.data['Close'])
        #mpf.plot(self.data, volume = False, addplot = apdict, type = 'line', mav = (20))
        
test = Stock_Analyzer()
#test.pivot_point()
#test.plot_data([ 'P_P', 'S1', 'R1', 'S2', 'R2'])

#test.ema(9)
test.cmf()
test.plot_data(['CMF'])

"""
test.ema(9)
test.ema(12)
test.plot_data(['EMA_9','EMA_12'])
print(test.data)
"""