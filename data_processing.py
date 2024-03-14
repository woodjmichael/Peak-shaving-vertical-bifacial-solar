__version__ = '0.0.2'


from os import listdir
from configparser import ConfigParser
from numpy import array, arange, nan, std, mean, sin, concatenate, argmax, pi, fft, empty, logical_and, abs, corrcoef, exp
import numpy as np
from pandas import DataFrame, infer_freq, read_csv, read_excel, date_range, concat, options, Timestamp, Timedelta, to_datetime, concat
from scipy.optimize import curve_fit
#from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

import math
import pandas as pd
import plotly.graph_objects as go
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots  

# Versions
# 1.0 - option to define dir

"""
Notes

1. try define all the right inputs to DataClass() and import_data(), otherwise make a good guess
2. no need for a config file now
3. all this code assumed the raw input data is in kw or kwh (otherwise convert manually)
4. for the year profile, there is untested code at the end of the file.. or do it manually
5. for me "df" (dataframe) is real measured data (but maybe cleaned)
    - whereas "profile" includes estimations (e.g. we dont have data for 9/2022 but it could be..)
"""

#
#
# Classes
#
#

class Data:
    def __init__(me, name, interval, timezone, max_export=None, main_breaker_rating_A=None, n_phases=3, voltage_ln_V=120, theme='plotly_white', egauge=False, dir='', config_file=True, units='kW'):
        options.plotting.backend = 'plotly' 
        me.theme = theme
        me.name = name
        me.interval = me.interval_lookup(interval) # prefer to use df.index.freq
        me.interval_h = {'1S':1/3600, '1T':1/60, '15T':0.25, '1H':1, '6H':6}[me.interval]
        me.timezone_name = timezone
        me.timezone = me.timezone_lookup(timezone)
        me.df = DataFrame([])
        me.units = units
        me.rolling_stdev = []
        me.rolling_mean = []
        me.np = nan # best naive persistence according to autocorrelation
        me.nan = nan # convenient to have
        me.profile = DataFrame( [], 
                                index=date_range(   start='2022-1-1 0:00', 
                                                    end='2022-12-31 23:45',
                                                    freq='15T'              ),
                                columns=['Load']                                 )

        if not isinstance(max_export, type(None)):                                                                        
            me.max_export = max_export # kW
        if main_breaker_rating_A:            
            me.max_import_1s = n_phases*voltage_ln_V*main_breaker_rating_A/1000 * 12 # 12x rated current possible for 1s
            me.max_import_60s = n_phases*voltage_ln_V*main_breaker_rating_A/1000 * 2 # 2x rated current possible for 60s
            print(f'calculated max import 1 s: {me.max_import_1s} kW')
            print(f'calculated max import 60 s: {me.max_import_60s} kW')

        if config_file:
            config = ConfigParser()
            config.read('config.ini')       

            if egauge:
                me.number = config['EGAUGE_NUMBERS'][me.name]
                me.egauge_dir = config['DEFAULT']['EGAUGE_DIR'] + 'egauge' +  me.number + '/'
                me.files = listdir(me.egauge_dir)        


    def interval_lookup(me, interval):
        lookup = {'60S':'1T', '1MIN':'1T', '15MIN':'15T','60MIN':'1H'}
        interval = interval.replace(' ','').upper()
        if interval in lookup.keys():
            interval = lookup[interval]
        return interval
        
        
    def timezone_lookup(me, tz):
        #       human time          standard/winter time    daylight/summer time
        return {'PT':'US/Pacific',  'PST':'ETC/GMT+8',      'PDT':'ETC/GMT+7',
                'MT':'US/Mountain', 'MST':'Etc/GMT+7',      'MDT':'ETC/GMT+6',
                'CT':'US/Central',  'CST':'Etc/GMT+6',      'CDT':'ETC/GMT+5',
                'ET':'US/Central',  'EST':'Etc/GMT+5',      'EDT':'ETC/GMT+4',}[tz]
        
    def timezone_reverse_lookup(me, tz):
        return {'ETC/GMT+8':'PST',
                'Etc/GMT+7':'MST',
                'Etc/GMT+6':'CST',
                'Etc/GMT+5':'EST'}[tz]        
        

    def analyze_autocorr(me,ds,lags_h=[0.25, 1, 2, 24, 7*24],lags_range=None):
        ds = fillna(ds,0)
        output = 'autocorr:'
        if lags_range is not None:
            for i in arange(lags_range[0],lags_range[1],lags_range[2]):
                a = ds.values[i:]
                b = ds.values[:-i]   
                c = corrcoef(a,b)[0,1]
                output += f' {i}={c:.2f}\n'            
        else:
            for lag in lags_h:
                i = int(lag/me.interval_h)
                a = ds.values[i:]
                b = ds.values[:-i]   
                c = corrcoef(a,b)[0,1]
                output += f' {lag}h={c:.2f}\n'
        print(output)
        
    def autocorr(me,lags=[],lags_range=None,feature='Load',plot=False):
        ds = me.df[feature]
        lag_h, acf = [], []
        if lags_range is not None:
            lags = list(arange(lags_range[0],lags_range[1],lags_range[2])) + lags
        for lag in lags:
            acf.append(concat((ds,ds.shift(lag)),axis=1).dropna().corr().iloc[0,1])
            lag_h.append(lag*me.interval_h)
        # if plot:
        #     points_per_week = int(168/me.interval_h)
        #     #df = me.autocorr(lags_range=(1,points_per_week,1))
        #     acf_week,lagh_week = [], []
        #     for lag in list(arange(1,points_per_week,1)):
        #         acf_week.append(concat((ds,ds.shift(lag)),axis=1).dropna().corr().iloc[0,1])
        #         lagh_week.append(lag*me.interval_h)
        #     df = DataFrame({'lag(h)':lag_h,'acf':acf})
        #     df.index = df['lag(h)']
        #     df[['acf']].plot()
        #     plt.show()
        return DataFrame({'lags':lags,'lag(h)':lag_h,'acf':acf})


    def import_data(me, filename, index_col=0, source=None, cols=None, rename=None, fix_index=True, 
                    standardize_timezone = None, diff=False, parse_dates=True, kwh_to_kw=False, 
                    autocorr_h=None, excel=False, unixtime=False):
        if source == 'egauge':
            cols=['Date & Time', 'Usage [kWh]']
        elif source == 'egauge 1 min':
            cols=['Date_&_Time_formatted', 'Usage_[kW]']
        elif source == 'entech':
            cols=['Time', 'SEL Total Active Power', 'PM 8244 Active Power Total', 'MCM Present PV Output']
        
        if excel:
            me.df = read_excel( filename,
                                comment='#',
                                usecols=cols,
                                index_col=index_col,
                                parse_dates=parse_dates)
        else:
                me.df = read_csv(   filename,
                                comment='#',
                                usecols=cols,
                                index_col=index_col,
                                parse_dates=parse_dates,
                                infer_datetime_format=True  )  # may speed up 5-10x
            
        if unixtime:
            input('None! Untested. Hit enter to continue: ')
            me.df.index = to_datetime(me.df.index, unit='s', origin='unix')
    
        
        if me.df.index.inferred_freq is not None:
            me.df.index.freq = me.df.index.inferred_freq
        else:
            me.df = me.df.resample(me.interval).mean()
        print(f'df freq: {me.df.index.freqstr}')
        
        #me.df = me.df.tz_localize(me.timezone)                                                                         

        if fix_index:
            me.fix_index()
            
        if standardize_timezone:            
            me.df = me.df.tz_localize(me.timezone)                              # old timezone
            me.df = me.df.tz_convert(me.timezone_lookup(standardize_timezone))  # new timezone
            me.df = me.df.tz_localize(None)                                     # de-clutter .csv
            
            me.timezone_name = standardize_timezone
            me.timezone = me.timezone_lookup(standardize_timezone)
            
        if rename:
            me.df.columns = rename

        if diff:
            me.df = me.df.diff().dropna() # egauge data is sometimes cumulative                         

        if kwh_to_kw:
            me.df = me.df/me.interval_h # kw = kwh / h            

        if 'Load' not in me.df.columns:
            input('Note this change!! Dont automatically calculate load from grid and solar. Hit enter to continue:')
            if 0: me.df['Load'] = me.df.Grid + me.df.Solar

        if source == 'entech': 
            me.df.columns = ['Grid', 'DER', 'Solar']
            me.df['Grid'] = -1*me.df.Grid
            me.df['Load'] = me.df.Grid + me.df.DER
            me.df = me.df[['Load']]       
            
        me.df.index.rename('Datetime ' + me.timezone_reverse_lookup(me.timezone), inplace=True)                    

        print(f'length: {len(me.df)}')
        print(f'NaNs: {me.df.isna().sum()[0]}')

        if autocorr_h:
            me.analyze_autocorr(me.df.Load, autocorr_h)

        return me.df        


    def import_file(me, filename, source, fix_index=False):
        if source == 'egauge':
            cols=['Date_&_Time_formatted', 'Usage_[kW]']
        elif source == 'entech':
            cols=['Time', 'SEL Total Active Power', 'PM 8244 Active Power Total', 'MCM Present PV Output']

        me.df = read_csv(   filename,
                            comment='#',
                            usecols=cols,
                            index_col=0,
                            parse_dates=True)

        if source == 'egauge':
            me.df.columns = ['Usage']

        if source == 'entech': 
            me.df.columns = ['Grid', 'DER', 'Solar']
            me.df['Grid'] = -1*me.df['Grid']
            me.df['Usage'] = me.df['Grid'] + me.df['DER']
            me.df = me.df[['Usage']]

        if fix_index:
            me.fix_index()

        return me.df


    def import_and_concatenate_downloaded_files(me, save=False):
        all_data = []
        for filename in me.files:
            print(filename)
            try:
                df = me.import_file(me.egauge_dir + filename)
                all_data.append(df)
            except:
                print(f'there was a problem with {filename}')

        print(f'\files imported: {len(all_data)}')
        print(f'files with problems: {len(me.files) - len(all_data)}')

        if save:
            concat(all_data).to_csv(f'data/{me.name}.csv')

        return concat(all_data)



    def convert_egauge_accumulator(me, filename, import_columns):
        df = read_csv(   filename, 
                            parse_dates=True, 
                            index_col=0, 
                            comment='#',
                            usecols=import_columns.append('Date & Time'))

        df.columns = ['Casino Usage Accum [kWh]', 'Moccasin Trail Usage Accum [kWh]']     
        df = df.sort_index()
        df['Casino Usage [kWh]'] = df['Casino Usage Accum [kWh]'].diff()
        df['Moccasin Trail Usage [kWh]'] = df['Moccasin Trail Usage Accum [kWh]'].diff()

        df = df.dropna()

        df['Casino Usage [kW]'] = df['Casino Usage [kWh]']*60
        df['Moccasin Trail Usage [kW]'] = df['Moccasin Trail Usage [kWh]']*60

        df = df[['Casino Usage [kW]', 'Moccasin Trail Usage [kW]']] 

        df = df.resample('15T').mean()                        
        df.to_csv('casino egauge 220215-220508 15min.csv')   


    def plot_old(me, features=['Usage'], period=None, interval='H', nans=False, wide=1000, high=None):
            nans = nans and me.df.isna().sum()[0]

            if period:
                begin = period[0]
                end = period[1]
                df = me.df.loc[:, features].resample(interval).mean()[begin:end]            
            else:
                df = me.df.loc[:, features]        

            if nans:
                # note how this is backwards, just for plotting
                #   1 = an NaN in data
                #   NaN = there is a real number in the data
                df['NaNs'] = me.df.loc[:, features].isna()*1
                df.loc[me.df['NaNs'] == 0, 'NaNs'] = nan
                features.append('NaNs')
                #me.df['NaNs'][me.df['NaNs'] == 0] = nan 

            labels=dict(index='Datetime '+me.timezone, value='Power [kW]', variable='')
            fig = df.plot(template=me.theme, width=wide, height=high, labels=labels); fig.show()


    def plot(me, df, features=None, period=None, interval=None, nans=False, wide=1000, high=None):
        if not interval:
            interval = me.interval

        if not features:
            features = df.columns[0]

        nans = nans and df.isna().sum()[0]

        if period:
            begin = period[0]
            end = period[1]
            df = df.loc[:, features].resample(interval).mean()[begin:end]            
        else:
            df = df.loc[:, features]        

        if nans:
            # note how this is backwards, just for plotting
            #   1 = an NaN in data
            #   NaN = there is a real number in the data
            df['NaNs'] = df.loc[:, features].isna()*1
            df.loc[df['NaNs'] == 0, 'NaNs'] = nan
            features.append('NaNs')
            #me.df['NaNs'][me.df['NaNs'] == 0] = nan 

        labels=dict(index=f'Date & Time [{me.timezone}]', value='Power [kW]', variable='')
        fig = df.plot(template=me.theme, width=wide, height=high, labels=labels); fig.show()


    def plot_daily_min_max(me, period=None, feature=None, wide=1000, high=None):
        if not feature:
            feature = me.df.columns[0]
    
        plot_features=['Min', 'Mean', 'Max']
        
        df = DataFrame([], index=me.df.resample('D').mean().index)
        df[plot_features[0]] = me.df[feature].resample('D').min()
        df[plot_features[1]] = me.df[feature].resample('D').mean()
        df[plot_features[2]] = me.df[feature].resample('D').max()

        if period:
            begin = period[0]
            end = period[1]
            df = df.loc[:, plot_features].mean()[begin:end]            
        else:
            df = df.loc[:, plot_features]   

        labels=dict(index='Datetime '+me.timezone, value='Power [kW]', variable='')
        fig = df.plot(template=me.theme, width=wide, height=high, labels=labels); fig.show()            


    def plot_weeks_overlaid(me, df, feature=None, transparency=0.2, size=(20,8)):
        if not feature:
            feature = df.columns[0]
        
        ds = df.resample('1H').mean()[feature]

        all_weeks = []
        for n_week in range(52):
            new_week = list( ds[ds.index.isocalendar().week == n_week] )
            if len(new_week) == 168:
                all_weeks.append(new_week)        

        plt.figure(figsize = size)
        plt.ylabel('Power (kW)')
        plt.xlabel('Hour of Week')
        plt.xticks(arange(0, 169, 24))        
        plt.plot(array(all_weeks).T, alpha = transparency)
        plt.show()        

        print('still using pyplot for now')
        
    def plot_weeks_overlaid_v2(me, feature=None, transparency=0.2, size=(20,8), title='All Weeks'):
        if feature is None:
            feature = me.df.columns[0]
        
        ds = me.df.resample('1H').mean()[feature]

        all_weeks = []
        for n_week in range(52):
            new_week = list( ds[ds.index.isocalendar().week == n_week] )
            if len(new_week) == 168:
                all_weeks.append(new_week)        

        plt.figure(figsize = size)
        plt.ylabel('Power (kW)')
        plt.xlabel('Hour of Week')
        plt.xticks(np.arange(0, 169, 24))        
        plt.plot(np.array(all_weeks).T, alpha = transparency)
        plt.title(title)
        plt.show()        

        print('still using pyplot for now')


    def calc_rolling_stats(me, interval='H', step='1T', plot=False):
        if interval == 'H' and step == '1T'and me.interval == '1S':
            window = 3600
            step = 60

        begin = 0
        end = window
        std = []
        avg = []
        d = me.df.dropna().values

        while end < len(me.d):
            std.append( std(d[begin:end]) )
            avg.append( mean(d[begin:end]) )
            begin += step
            end += step


        if plot:
            plt.figure(figsize=(20,8))
            plt.plot(avg)
            plt.title(f'rolling hourly mean (mean of means {mean(avg):.2f})')
            plt.show()

            plt.figure(figsize=(20,8))
            plt.plot(std)
            plt.title(f'rolling hourly stdev (mean {mean(std):.2f})')
            plt.show()

        else:
            print(f'rolling hourly mean (mean of means {mean(avg):.2f})')
            print(f'rolling hourly stdev (mean {mean(std):.2f})')

        me.rolling_stdev = std
        me.rolling_mean = avg


    def show_nans(me):
        if me.df.empty:
            print('no data')
        else:
            print(me.df[me.df.isna().values])


    def fill_nans(me):
        if me.df.empty:
            print('no data')
        else:
            print(f'NaNs found {me.df.isna().sum()}')
            me.df = me.df.ffill()
            me.df = me.df.bfill() # in case the first value is nan

      
    def analyze_index(me, show_gaps=False):
        if me.df.empty:
            print('no data')
        else:
            print(f'begin: {me.df.index[0]}')
            print(f'end: {me.df.index[-1]}')
            gaps = me.df['dindex'][me.df['dindex'] > me.df['dindex'][1]]
            print(f'gaps: {gaps}')
            print('')
            print('intervals analysis:')
            me.df['dindex'] = me.df.index.to_series().diff()
            print(me.df['dindex'].describe())
            me.df = me.df.drop(columns=['dindex'])

        if show_gaps:
            print( me.df['dindex'][me.df['dindex'] > me.df['dindex'][1]].to_string() )


    def fix_index(me):        
        me.df = me.df.resample(me.interval).mean() # gives the index a designated frequency and removes duplicates

        start, end = me.df.index[0], me.df.index[-1]
        idx = date_range(start=start, end=end, freq=me.interval)
        temp = DataFrame(nan, index=idx, columns=me.df.columns)

        temp.loc[me.df.index] = me.df.values

        filled = len(temp) - len(me.df)
        print(f'indices filled: {filled} ({100*filled/len(temp):.2f} %) \n')
        me.df = temp.copy(deep=True) 


    def find_outliers(me, z_thresh=3, plot=True, wide=1000, feature='Load', diff=False, replace_with_nans=True):
        # dont work on the real data
        df = me.df.copy(deep=True)
        df['z-score'] = nan
        
        if diff:
            df['diff'] = df[feature].diff()
            df['diff'].iloc[0] = 0

        # loop through each hour of each weekday and calculate z-score
        for d in range(7):
            for h in range(24):
                day_hour = logical_and(df.index.weekday == d, df.index.hour == h)
                mean_dh = df[feature][day_hour].mean()
                std_dh = df[feature][day_hour].std()
                df.loc[day_hour, 'z-score'] = abs( (df.loc[day_hour, feature] - mean_dh) / std_dh )
                
                if diff:
                    mean_dh_diff = df['diff'][day_hour].mean()
                    std_dh_diff = df['diff'][day_hour].std()
                    df.loc[day_hour, 'diff z-score'] = abs( (df.loc[day_hour, 'diff'] - mean_dh_diff) / std_dh_diff )

        
        # delete outliers in real data
        if replace_with_nans:
            n = len(df[df['z-score']>z_thresh])
            me.df[df['z-score']>z_thresh] = nan
            print(f'outliers found: {n}')
        
        if diff:
            me.df[df['diff z-score']>z_thresh] = nan
            n = n + len(df[df['diff z-score']>z_thresh])
            print(f'diff outliers found: {n}')

        # show
        if plot:
            df['Outliers'] = nan
            
            print( df[[feature,'z-score']][df['z-score']>z_thresh].sort_values(by='z-score'), '\n')               
            df['Outliers'][df['z-score']>z_thresh] = df[feature][df['z-score']>z_thresh]

            if diff:
                df['Outliers (diff)'] = nan
                print( df[[feature,'diff z-score']][df['diff z-score']>z_thresh].sort_values(by='diff z-score'), '\n')
                df['Outliers (diff)'][df['diff z-score']>z_thresh] = df[feature][df['diff z-score']>z_thresh]

            #me.plot(df, features=['Load, Outliers'])
            if not diff:
                fig = df[[feature, 'Outliers']].plot(template=me.theme, width=wide); fig.show()                  
            elif diff:
                fig = df[[feature, 'Outliers', 'Outliers (diff)']].plot(template=me.theme, width=wide); fig.show()  
        
        return df                


    def replace_nans_with_persistence(me, lags=None,lags_h=None,
                                      lags_range_h=None,lags_multiple_ranges_h=None, feature='Load'):
        
        if lags_multiple_ranges_h is not None:
            lags = []
            for lags_range in lags_multiple_ranges_h:
                lags = lags + list(arange(lags_range[0], lags_range[1], lags_range[2]))
        elif lags_range_h is not None:
            lags = list(arange(lags_range_h[0], lags_range_h[1], lags_range_h[2])) + lags_h
        elif lags_h is not None:
            lags = lags_h
        else:
            lags = [x*me.interval_h for x in lags]

        for h in lags: # vector of naive persistence lags to search
            # save the current list of nans
            df_nans = me.df.loc[me.df.isna().values, feature]

            # loop through all (remaining) nans
            for i in range(len(df_nans)):

                # for the index of each nan
                idx = df_nans.index[i]
   
                # look for a replacement value at location -np
                try: 
                    if me.df.loc[idx - Timedelta(hours=h), feature] is not nan: # if its nan, dont bother
                        me.df.loc[idx, feature] = me.df.loc[idx - Timedelta(hours=h), feature]                
                # or at +np (may be near beginning of dataset)
                except:
                    if me.df.loc[idx + Timedelta(hours=h), feature] is not nan: # if its nan, dont bother
                        me.df.loc[idx, feature] = me.df.loc[idx + Timedelta(hours=h), feature]   

        print(f'NaNs after replacement: {me.df.isna().sum()[0]} \n')
              

    def omit_physically_impossible_outliers(me):
        if not isinstance(me.max_export,type(None)) \
            and not isinstance(me.max_import_1s,type(None)) \
            and not isinstance(me.max_import_60s,type(None)):

            print(f'number of impossibles (hi): {len(me.df[me.df.values > me.max_import_1s])}')
            print(f'number of impossibles (lo): {len(me.df[me.df.values < -1*me.max_export])}')

            me.df[me.df.values > me.max_import_1s] = nan
            me.df[me.df.values < -1*me.max_export] = nan

        if not me.max_import_1s:
            print('no main breaker size defined')            


    def omit_statistically_improbable_outliers(me):
        # needs work
        if 0:
            stdev = std(me.df.dropna().values)
            avg = mean(me.df.dropna().values)
            lim_hi = avg + 3*stdev
            lim_lo = avg - 3*stdev

            print(f'number of values too hi: {len(me.df[me.df.values > lim_hi])}')
            print(f'number of values too lo: {len(me.df[me.df.values < lim_lo])}')
            
            me.df[me.df.values > lim_hi]


    def fit_sin(me, tt, yy):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = array(tt)
        yy = array(yy)
        ff = fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(fft.fft(yy))
        guess_freq = abs(ff[argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = std(yy) * 2.**0.5
        guess_offset = mean(yy)
        guess = array([guess_amp, 2.*pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c):  return A * sin(w*t + p) + c
        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*pi)
        fitfunc = lambda t: A * sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": max(pcov), "rawres": (guess,popt,pcov)}


    def latest_good_data(me, ds):
        if ds.isna().sum() > 0:
            ts = ds[ds.isna()].index.min() - Timedelta(me.interval)
        else:
            ts = ds.index.max()
        return ts

    def trim_partial_days(me, df, align_first_weekday=False):


        y1 = df.index.min().year
        m1 = df.index.min().month
        d1 = df.index.min().day + 1
        first_full_day = f'{y1}-{m1}-{d1}'

        y2 = df.index.max().year
        m2 = df.index.max().month
        d2 = df.index.max().day - 1
        last_full_day = f'{y2}-{m2}-{d2}'

        if align_first_weekday:
            begin_weekday = me.profile.index.min().weekday()            
            some_first_weekday = df.loc[df.index.weekday == begin_weekday].index.min()
            df = df.loc[some_first_weekday:, :]

        return df.loc[first_full_day : last_full_day, :]        


    def profile_estimation(me, estimation_periods=None, spring_split = 3, summer_split = 5, fall_split = 9, winter_split=12):
        # only works when data comes from one single year

        # actually we could choose this since its just a profile, not really 2022

        # data we will use to estimate the periods where we dont have data
        if not estimation_periods:
            winter =    me.df.loc[:'2022-3-31', :]
            shoulder =  me.df.loc['2022-4-1':'2022-4-26', :]
            summer =    me.df.loc['2022-4-27':, :]
        else:
            (winter, shoulder, summer) = estimation_periods

        # begin the loop
        estimations = me.trim_partial_days(winter, align_first_weekday=True)
        begin, end = 0, len(estimations)

        while me.latest_good_data(me.profile['Load']) < Timestamp('2022-12-31 23:45'):  

            me.profile['Load'].iloc[begin:end] = estimations['Load'].values[:(end-begin)]

            print(f'filled {begin} to {end} up to', me.latest_good_data(me.profile['Load']) )

            latest_good_month = me.latest_good_data(me.profile['Load']).month
            if latest_good_month >= spring_split:
                estimations = me.trim_partial_days(shoulder)
            if latest_good_month >= summer_split:
                estimations = me.trim_partial_days(summer)
            if latest_good_month >= fall_split:
                estimations = me.trim_partial_days(shoulder)
            if latest_good_month >= winter_split:
                estimations = me.trim_partial_days(winter)

            begin = end
            end = min(end+len(estimations), 35040)

        # overwrite some estimations with real measurements
        meas = me.trim_partial_days(me.df)
        me.profile.loc[meas.index, 'Load'] = meas.loc[:, 'Load']  

        me.profile = me.profile.astype('float') # describe() was showing data to be of type "object"

        print('')

        return me.profile


    def scale_estimations(me, energy_targets):
        scales = 'scaling factors: '
        demands = 'demands (scaled) kw: '

        for month in range(1,13):

            # vector
            idx_month = me.profile.index.month == month

            # actual energy
            energy = me.profile['Load'][idx_month].sum()*me.interval_h
            scale = energy_targets[month-1]/energy

            # scale and put into profile
            me.profile['Load'][idx_month] = scale * me.profile['Load'][idx_month].values

            # track scaling factors and demand
            scales += (f'{scale:.3f} ')
            demand = me.profile['Load'][idx_month].max()
            demands += (f'{demand:.0f} ')

        print(scales)
        print(demands)    

   
    def replaceNANsWithPersistence(me, lags_h=[24, 168], lags_range_h=None, lags_multiple_ranges_h=None, feature='Load kW'):
        errors = {'count':0,'no td':[],'no valid':[]}
        
        if lags_multiple_ranges_h is not None:
            lags = []
            for lags_range in lags_multiple_ranges_h:
                lags = lags + list(arange(lags_range[0], lags_range[1], lags_range[2]))
        elif lags_range_h is not None:
            lags = list(arange(lags_range_h[0], lags_range_h[1], lags_range_h[2])) + lags_h
        else:
            lags = lags_h

        # save the current list of nans
        nans = me.df.loc[me.df.isna().values, feature].index

        # loop through all nans
        for idx in nans.sort_values(ascending=False):            
            try:
                lag_iter = iter(lags)
                lag = next(lag_iter)
                while (idx - Timedelta(hours=lag)) in nans:
                    lag = next(lag_iter)
                me.df.loc[idx,feature] = me.df.loc[idx - Timedelta(hours=lag),feature]
                #print('Replace',idx,'with',idx-Timedelta(hours=lag),me.df.Load[idx])
            except:
                print('No time deterministic lags',idx)
                errors['count'] = errors['count'] + 1
                errors['no td'].append(idx)
                try:
                    lag_iter = iter(lags)
                    lag = next(lag_iter)
                    while (idx + Timedelta(hours=lag)) in nans:
                        lag = next(lag_iter)
                    me.df.loc[idx,feature] = me.df.loc[idx + Timedelta(hours=lag),feature]
                except:
                    me.df.loc[idx,feature] = nan
                    print('No valid lags',idx)
                    errors['count'] = errors['count'] + 1
                    errors['no valid'].append(idx)
                    
        print(f'NaNs after replacement: {me.df.isna().sum()[0]} \n')                    
        return errors    
                
        # for lag in lags: # vector of naive persistence lags to search
        #         # look for a replacement value at location -np
        #         try: 
        #             if me.df.loc[idx - Timedelta(hours=lag), 'Load'] is not nan: # if its nan, dont bother
        #                 me.df.loc[idx, 'Load'] = me.df.loc[idx - Timedelta(hours=lag), 'Load']
        #                 print('Replace')
        #                 print('       ',idx,me.df.loc[idx,'Load'])
        #                 print(' with')
        #                 print('       ',idx-Timedelta(hours=lag), me.df.loc[idx - Timedelta(hours=lag), 'Load'])

        #         # or at +np (may be near beginning of dataset)
        #         except:
        #             if me.df.loc[idx + Timedelta(hours=lag), 'Load'] is not nan: # if its nan, dont bother
        #                 me.df.loc[idx, 'Load'] = me.df.loc[idx + Timedelta(hours=lag), 'Load']   
        #                 print('Replace',idx,'with',idx+Timedelta(hours=lag))

        # print(f'NaNs after replacement: {me.df.isna().sum()[0]} \n')                    

    def import_and_clean_openweather_historical_forecast(me, filename, feature, start=None, end=None):
        df = read_csv(filename,
                      index_col='forecast dt unixtime',
                      usecols=['forecast dt unixtime','slice dt unixtime',feature])
        
        # convert unix times to datetimes
        df.index = to_datetime(df.index, unit='s', origin='unix')
        df['slice dt unixtime'] = to_datetime(df['slice dt unixtime'], unit='s', origin='unix')
        df.columns = ['Forecasted dt',feature]
        
        # output file index
        if not start:
            start = df.index[0]
        if not end:
            end = df.index[-1]
        idx = date_range(start=start,end=end,freq=me.interval)
        
        # easier to work with the forecast horizon delta (e.g. t+6h)
        df['deltaHorizon'] = df['Forecasted dt'] - df.index.to_series()
        
        # make vector data 2D, while checking that the deltas are same
        data = []
        deltaHorizon0 = df.loc[idx[0], 'deltaHorizon'].values
        for t in idx:
            assert sum(df.loc[t,'deltaHorizon'].values - deltaHorizon0) == 0
            data.append(df.loc[t,feature].to_list())
            
        # fill new df
        df = DataFrame(data, index=idx)
        df.index = df.index.rename(f'Datetime {me.timezone_name}')
        feature = feature.capitalize()
        df.columns = [f'{feature} Hourly.{h}' for h in df.columns]
    
        me.df2d = df.copy(deep=True)
        
        
#
#
# Functions
#
#       

def fillna(ds:pd.Series):
    return ds.replace(np.nan, 0)
        
def order_of_magnitude(x:float)->int:
    """ Calculate order of magnitude

    Args:
        x (float): real number

    Returns:
        int: order of magnitude
    """
    
    return math.floor(math.log(x, 10))


def upsample_ffill(df,periods,freq):
    df2 = pd.DataFrame([],index=pd.date_range(df.index[0],periods=periods,freq=freq))
    for col in df.columns:
        df2.loc[df.index,col] = df[col].values
    df2 = df2.ffill()
    return df2


def shift_and_wraparound(ds:pd.Series,i:int):
    """ Shift data and wraparound the end
    pos = shift left/up
    neg = shift right/down
    """
    return list(ds.values[i:]) + list(ds.values[:i])
    
    
def calc_monthly_peaks(ds:pd.Series,peak_begin:int,peak_end:int) -> pd.Series:
    """ Calculate max power (1 hr non-moving average) of the month

    Args:
        ds (pd.Series): timeseries to calculate on
        peak_begin (int): start of peak TOU period (inclusive)
        peak_end (int): end of peak TOU period (exclusive)

    Returns:
        pd.Series: _description_
    """
    r = pd.DataFrame([],columns=[f'{ds.name} kw',f'{ds.name} t'])
    for year in ds.index.year.unique():
        ds_year = ds[ds.index.year==year]
        for month in ds_year.index.month.unique():
            ds_month = ds_year[ds_year.index.month==month]
            idx = [peak_begin<=h<peak_end for h in ds_month.index.hour]
            r.loc[len(r)] = [round(ds_month[idx].max(),1),
                             ds_month[idx].idxmax()]
    return r


def plot_daily(ds:pd.Series,
               interval_min:int,
               alpha:float=0.1,
               title:str=None,
               boxplot:bool=False):
    """ Plot a series with days super-imposed on each other. Index should be complete (no gaps)
    for this to work right. Trims any remaining data after an integer number of days.

    Args:
        ds (pd.Series): pandas series to plot
        interval_min (int): timeseries data interval
        alpha (float, optional): transparency of plot lines, defaults to 0.1
        begin_on_monday (bool, optional): have the first day on the plot be monday, defaults to True
    """
    dpd = int(24*60/interval_min) # data per day
    ds2 = ds.copy(deep=True)
    dt_start = ds.index.min()
    if dt_start != dt_start.floor('1d'):
        dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
    ds2 = ds2[dt_start:]
    n_days = len(ds2)//(dpd)
    ds2 = ds2.iloc[:int(n_days*dpd)]
    
    #t = list(range(24)) # hours
    plt.plot(ds2.values.reshape(n_days,dpd).T,alpha=alpha)
    plt.ylabel(ds.name)
    plt.xlabel('Hours from 0:00')
    plt.title(title)
    plt.show()
    
    if boxplot:
        cols = []
        cols = [f'{h}:00' for h in range(24)]
        pd.DataFrame(ds2.values.reshape(n_days,dpd).T,columns=cols).boxplot()

def plot_daily_2(ds,
                title:str=None,
                ylabel:str=None,
                alpha:float=0.1,
                colors:list=['indigo','gold','magenta']):
    """ Plot a series with days super-imposed on each other. Index should be complete (no gaps)
    for this to work right. Trims any remaining data after an integer number of weeks.

    Args:
        ds (pd.Series or list): pandas series(es) to plot
        title (str, optional): title
        ylabel (str, optional): what to call the y-axis        
        interval_min (int): timeseries data interval
        alpha (float, optional): transparency of plot lines, defaults to 0.1
        colors (list, optional): list of colors strings
    """
    if not isinstance(ds,(list,tuple)):
        ylabel = ds.name
        ds = [ds]
        
    interval_min = int(ds[0].index.to_series().diff().mean().seconds/60)
    dpd = int(24*60/interval_min) # data per day
    plt.figure(figsize=(10,5))
    t = [x/dpd for x in range(dpd)] # days    

    for ds2,color in zip(ds,colors):
        ds2 = ds2.copy(deep=True)
        dt_start = ds2.index.min()
        if dt_start != dt_start.floor('1d'):
            dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
        ds2 = ds2[dt_start:]
        n_days = len(ds2)//(dpd)
        ds2 = ds2.iloc[:int(n_days*dpd)]
        if len(ds)>1:
            plt.plot(t,ds2.values.reshape(n_days,dpd).T,color,alpha=alpha)
        else:
            plt.plot(t,ds2.values.reshape(n_days,dpd).T,alpha=alpha)
    plt.ylabel(ylabel)
    #plt.xlabel('Days from Monday 0:00')
    plt.title(title)
    if len(ds)>1:
        legend_items = []
        for s,color in zip(ds,colors):
            legend_items.append(mpatches.Patch(color=color, label=s.name))
        plt.legend(handles=legend_items)
    plt.show()


def plot_weekly(ds,
                title:str=None,
                ylabel:str=None,
                alpha:float=0.1,
                begin_on_monday:bool=True,
                colors:list=['indigo','gold','magenta']):
    """ Plot a series with weeks super-imposed on each other. Index should be complete (no gaps)
    for this to work right. Trims any remaining data after an integer number of weeks.

    Args:
        ds (pd.Series or list): pandas series(es) to plot
        title (str, optional): title
        ylabel (str, optional): what to call the y-axis        
        interval_min (int): timeseries data interval
        alpha (float, optional): transparency of plot lines, defaults to 0.1
        begin_on_monday (bool, optional): have the first day on the plot be monday, defaults to True
        colors (list, optional): list of colors strings
    """
    if not isinstance(ds,(list,tuple)):
        ylabel = ds.name
        ds = [ds]
        
    interval_min = int(ds[0].index.to_series().diff().mean().seconds/60)
    dpd = int(24*60/interval_min) # data per day
    plt.figure(figsize=(10,5))
    t = [x/dpd for x in range(7*dpd)] # days    

    for ds2,color in zip(ds,colors):
        ds2 = ds2.copy(deep=True)
        dt_start = ds2.index.min()
        if dt_start != dt_start.floor('1d'):
            dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
        if begin_on_monday and (dt_start.weekday() != 0):
            days = 7 - dt_start.weekday()
            dt_start += pd.Timedelta(hours=24*days)
        ds2 = ds2[dt_start:]
        n_weeks = len(ds2)//(7*dpd)
        ds2 = ds2.iloc[:int(n_weeks*7*dpd)]
        if len(ds)>1:
            plt.plot(t,ds2.values.reshape(n_weeks,7*dpd).T,color,alpha=alpha)
        else:
            plt.plot(t,ds2.values.reshape(n_weeks,7*dpd).T,alpha=alpha)
    plt.ylabel(ylabel)
    plt.xlabel('Days from Monday 0:00')
    plt.title(title)
    if len(ds)>1:
        legend_items = []
        for s,color in zip(ds,colors):
            legend_items.append(mpatches.Patch(color=color, label=s.name))
        plt.legend(handles=legend_items,loc='upper right')
    plt.show()    

def plot_weekly_3(ds,
                title:str=None,
                ylabel:str=None,
                alpha:float=0.1,
                begin_on_monday:bool=True,
                colors:list=['indigo','gold','magenta']):
    """ Plot a series with weeks super-imposed on each other. Index should be complete (no gaps)
    for this to work right. Trims any remaining data after an integer number of weeks.

    Args:
        ds (pd.Series or list): pandas series(es) to plot
        title (str, optional): title
        ylabel (str, optional): what to call the y-axis        
        interval_min (int): timeseries data interval
        alpha (float, optional): transparency of plot lines, defaults to 0.1
        begin_on_monday (bool, optional): have the first day on the plot be monday, defaults to True
        colors (list, optional): list of colors strings
    """
    if not isinstance(ds,(list,tuple)):
        ylabel = ds.name
        ds = [ds]
        
    interval_min = int(ds[0].index.to_series().diff().mean().seconds/60)
    dpd = int(24*60/interval_min) # data per day
    plt.figure(figsize=(10,5))
    t = [x/dpd for x in range(7*dpd)] # days    

    for ds2,color in zip(ds,colors):
        ds2 = ds2.copy(deep=True)
        dt_start = ds2.index.min()
        if dt_start != dt_start.floor('1d'):
            dt_start = dt_start.floor('1d') + pd.Timedelta(hours=24)
        if begin_on_monday and (dt_start.weekday() != 0):
            days = 7 - dt_start.weekday()
            dt_start += pd.Timedelta(hours=24*days)
        ds2 = ds2[dt_start:]
        n_weeks = len(ds2)//(7*dpd)
        ds2 = ds2.iloc[:int(n_weeks*7*dpd)]
        if len(ds)>1:
            plt.plot(t,ds2.values.reshape(n_weeks,7*dpd).T,color,alpha=alpha)
        else:
            plt.plot(t,ds2.values.reshape(n_weeks,7*dpd).T,alpha=alpha)
    plt.ylabel(ylabel)
    plt.xlabel('Days from Monday 0:00')
    plt.title(title)
    if len(ds)>1:
        legend_items = []
        for s,color in zip(ds,colors):
            legend_items.append(mpatches.Patch(color=color, label=s.name))
        plt.legend(handles=legend_items)
    plt.show()

def plotly_stacked(_df:pd.DataFrame,
                   solar='solar',
                   solar_name='Solar',
                   load='load',
                   load_name='Load',
                   batt='batt',
                   discharge='discharge',
                   discharge_name='Battery Discharge',
                   charge='charge',
                   load_charge_name='Load + Battery Charge',
                   utility='utility',
                   utility_name='Import',        
                   soc='soc',
                   soc_name='SOC (right axis)',
                   soe='soe',
                   soe_name='SOE (right axis)',
                   threshold0=None,
                   threshold0_h=None,
                   threshold0_name = 'Threshold 0',
                   threshold1=None,
                   threshold1_h=None,
                   threshold1_name = 'Threshold 1',
                   threshold2=None,
                   threshold2_h=None,
                   threshold2_name = 'Threshold 2',
                   ylim=None,
                   size=None,
                   title=None,
                   fig=None,
                   units_power='kW',
                   units_energy='kWh',
                   round_digits=1,
                   upsample_min=None,
                   template='plotly_white'):
    """ Make plotly graph with some data stacked in area-fill style.
    
    Template options are :['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']
    """
    
    df = _df.copy(deep=True) # we'll be modifying this
    
    # upsample for more accurate viewing
    if upsample_min is not None:
        freq_min = int(df.index.to_series().diff().dropna().mean().seconds/60)
        new_length = len(df) * (freq_min / upsample_min)
        df = upsample_ffill(df,freq=f'{upsample_min}min',periods=new_length)
        
    # threshold vectors
    if threshold0 is not None:
        df['threshold0'] = [threshold0 if x in threshold0_h else pd.NA for x in df.index.hour]
    if threshold1 is not None:
        df['threshold1'] = [threshold1 if x in threshold1_h else pd.NA for x in df.index.hour]
    if threshold2 is not None:
        df['threshold2'] = [threshold2 if x in threshold2_h else pd.NA for x in df.index.hour]
    
    #export='export'
    loadPlusCharge = 'loadPlusCharge'

    if charge not in df.columns:
        df[charge] =    [max(0,-1*x) for x in df[batt]]
        df[discharge] =    [max(0,x) for x in df[batt]]    
    df[loadPlusCharge] = df[load]+df[charge]
    #df[export] = df[solar] - df[loadPlusCharge] #[-1*min(0,x) for x in df[utility]]
    df[utility] = [max(0,x) for x in df[utility]]
    df[solar] = df[solar]#df[load] - df[utility]
    
    # plot
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            name=utility_name,
            x=df.index, y=df[utility].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='darkseagreen'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=solar_name,
            x=df.index, y=df[solar].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0,color='gold'),
        ),
        secondary_y=False,
    )
    # fig.add_trace(
    #     go.Scatter(
    #         name='Export',
    #         x=df.index, y=df[export],
    #         mode='lines',
    #         stackgroup='one',
    #         line=dict(width=0,color='khaki'),
    #     ),
    #     secondary_y=False,
    # )
    fig.add_trace(
        go.Scatter(
            name=discharge_name,
            x=df.index, y=df[discharge].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='dodgerblue'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_charge_name,
            x=df.index, y=df[loadPlusCharge].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5,
                      #dash='dash',
                      color='dodgerblue',
                      #color='mediumvioletred',
                      ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_name,
            x=df.index, y=df[load].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5,
                      #color='indigo'),
                      color='mediumvioletred',)
        ),
        secondary_y=False,
    )
    if threshold0 is not None:
        if threshold1 is None:
            name = 'Threshold'
        else:
            name = threshold0_name
        fig.add_trace(
            go.Scatter(
                name=name,
                x=df.index, y=df['threshold0'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='palevioletred',
                          color='gray',
                          dash='longdash', # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
                          ),
            ),
            secondary_y=False,
        )
    if threshold1 is not None:
        fig.add_trace(
            go.Scatter(
                name=threshold1_name,
                x=df.index, y=df['threshold1'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='mediumvioletred'),
                          color='black',
                          dash='dash',
                          ),
            ),
            secondary_y=False,
        )
    if threshold2 is not None:
        fig.add_trace(
            go.Scatter(
                name=threshold2_name,
                x=df.index, y=df['threshold2'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='crimson'),
                          color='black',
                          dash='dot',
                          ),
            ),
            secondary_y=False,
        )        
    if soc in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soc_name,
                x=df.index, y=(df[soc]*100).round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dashdot',color='coral'),
            ),
            secondary_y=True,
        ) 
    elif soe in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soe_name,
                x=df.index, y=df[soe].round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dashdot',color='coral'),
            ),
            secondary_y=True,
        )
           
    fig.update_traces(hovertemplate=None)#, xhoverformat='%{4}f')
    fig.update_layout(hovermode='x',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h'),
                      legend_traceorder='reversed',
                      template=template,
                      title=title,)
    if size is not None:
        fig.update_layout(width=size[0],height=size[1])

        fig.update_yaxes(title_text=units_power, secondary_y=False)
    
    if soc in df.columns:
        fig.update_yaxes(title_text='%',range=(0, 100),secondary_y=True)
    elif soe in df.columns:
        fig.update_yaxes(title_text=units_energy,range=(0, df[soe].max()),secondary_y=True)

    if ylim is None:
        ymax = max(df[loadPlusCharge].max(),df[utility].max(),df[solar].max())
        fig.update_yaxes(range=(-.025*ymax, 1.1*ymax),secondary_y=False)
    else:
        fig.update_yaxes(range=(ylim[0], ylim[1]),secondary_y=False)
        
    fig.show()
    
def plotly_stacked_4tou(_df:pd.DataFrame,
                   solar='solar',
                   solar_name='Solar',
                   load='load',
                   load_name='Load',
                   batt='batt',
                   discharge='discharge',
                   discharge_name='Battery Discharge',
                   charge='charge',
                   load_charge_name='Load + Battery Charge',
                   utility='utility',
                   utility_name='Import',        
                   soc='soc',
                   soc_name='SOC (right axis)',
                   soe='soe',
                   soe_name='SOE (right axis)',
                   threshold0=None,
                   threshold0_h=None,
                   threshold0_name = 'Threshold 0',
                   threshold1=None,
                   threshold1_h=None,
                   threshold1_name = 'Threshold 1',
                   threshold2=None,
                   threshold2_h=None,
                   threshold2_name = 'Threshold 2',
                   threshold3=None,
                   threshold3_h=None,
                   threshold3_name = 'Threshold 3',
                   ylim=None,
                   size=None,
                   title=None,
                   fig=None,
                   units_power='kW',
                   units_energy='kWh',
                   round_digits=1,
                   upsample_min=None,
                   template='plotly_white',
                   save_path=None):
    """ Make plotly graph with some data stacked in area-fill style.
    
    Template options are :['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']
    """
    
    df = _df.copy(deep=True) # we'll be modifying this
    
    # upsample for more accurate viewing
    if upsample_min is not None:
        freq_min = int(df.index.to_series().diff().dropna().mean().seconds/60)
        new_length = len(df) * (freq_min / upsample_min)
        df = upsample_ffill(df,freq=f'{upsample_min}min',periods=new_length)
        
    # threshold vectors
    if threshold0 is not None:
        df['threshold0'] = [threshold0 if x in threshold0_h else pd.NA for x in df.index.hour]
    if threshold1 is not None:
        df['threshold1'] = [threshold1 if x in threshold1_h else pd.NA for x in df.index.hour]
    if threshold2 is not None:
        df['threshold2'] = [threshold2 if x in threshold2_h else pd.NA for x in df.index.hour]
    if threshold3 is not None:
        df['threshold3'] = [threshold3 if x in threshold3_h else pd.NA for x in df.index.hour]
    
    #export='export'
    loadPlusCharge = 'loadPlusCharge'

    if charge not in df.columns:
        df[charge] =    [max(0,-1*x) for x in df[batt]]
        df[discharge] =    [max(0,x) for x in df[batt]]    
    df[loadPlusCharge] = df[load]+df[charge]
    #df[export] = df[solar] - df[loadPlusCharge] #[-1*min(0,x) for x in df[utility]]
    df[utility] = [max(0,x) for x in df[utility]]
    df[solar] = df[solar]#df[load] - df[utility]
    
    # plot
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            name=utility_name,
            x=df.index, y=df[utility].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='darkseagreen'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=solar_name,
            x=df.index, y=df[solar].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0,color='gold'),
        ),
        secondary_y=False,
    )
    # fig.add_trace(
    #     go.Scatter(
    #         name='Export',
    #         x=df.index, y=df[export],
    #         mode='lines',
    #         stackgroup='one',
    #         line=dict(width=0,color='khaki'),
    #     ),
    #     secondary_y=False,
    # )
    fig.add_trace(
        go.Scatter(
            name=discharge_name,
            x=df.index, y=df[discharge].round(round_digits),
            mode='lines',
            stackgroup='one',
            line=dict(width=0, color='dodgerblue'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_charge_name,
            x=df.index, y=df[loadPlusCharge].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5,
                      #dash='dash',
                      color='dodgerblue',
                      #color='mediumvioletred',
                      ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=load_name,
            x=df.index, y=df[load].round(round_digits),
            mode='lines',
            #stackgroup='one',
            line=dict(width=1.5,
                      #color='indigo'),
                      color='mediumvioletred',)
        ),
        secondary_y=False,
    )
    if threshold0 is not None:
        if threshold1 is None:
            name = 'Threshold'
        else:
            name = threshold0_name
        fig.add_trace(
            go.Scatter(
                name=name,
                x=df.index, y=df['threshold0'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='palevioletred',
                          color='gray',
                          dash='longdash', # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
                          ),
            ),
            secondary_y=False,
        )
    if threshold1 is not None:
        fig.add_trace(
            go.Scatter(
                name=threshold1_name,
                x=df.index, y=df['threshold1'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='mediumvioletred'),
                          color='black',
                          dash='dash',
                          ),
            ),
            secondary_y=False,
        )
    if threshold2 is not None:
        fig.add_trace(
            go.Scatter(
                name=threshold2_name,
                x=df.index, y=df['threshold2'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='crimson'),
                          color='black',
                          dash='dot',
                          ),
            ),
            secondary_y=False,
        )
    if threshold3 is not None:
        fig.add_trace(
            go.Scatter(
                name=threshold3_name,
                x=df.index, y=df['threshold3'],
                mode='lines',
                #stackgroup='one',
                line=dict(width=3,
                          #color='crimson'),
                          color='black',
                          dash='dot',
                          ),
            ),
            secondary_y=False,
        )        
    if soc in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soc_name,
                x=df.index, y=(df[soc]*100).round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dashdot',color='coral'),
            ),
            secondary_y=True,
        ) 
    elif soe in df.columns:
        fig.add_trace(
            go.Scatter(
                name=soe_name,
                x=df.index, y=df[soe].round(round_digits),
                mode='lines',
                line=dict(width=1, dash='dashdot',color='coral'),
            ),
            secondary_y=True,
        )
           
    fig.update_traces(hovertemplate=None)#, xhoverformat='%{4}f')
    fig.update_layout(hovermode='x',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h'),
                      legend_traceorder='reversed',
                      template=template,
                      title=title,)
    if size is not None:
        fig.update_layout(width=size[0],height=size[1])

        fig.update_yaxes(title_text=units_power, secondary_y=False)
    
    if soc in df.columns:
        fig.update_yaxes(title_text='%',range=(0, 100),secondary_y=True)
    elif soe in df.columns:
        fig.update_yaxes(title_text=units_energy,range=(0, df[soe].max()),secondary_y=True)

    if ylim is None:
        ymax = max(df[loadPlusCharge].max(),df[utility].max(),df[solar].max())
        fig.update_yaxes(range=(-.025*ymax, 1.1*ymax),secondary_y=False)
    else:
        fig.update_yaxes(range=(ylim[0], ylim[1]),secondary_y=False)
        
    fig.show()
    
    if save_path is not None:
        fig.write_image(save_path)    
    
#
#
# Main
#
#
    


if __name__ == '__main__':    
    data =      Data('badriver_clinic', 
                     interval='1 h', 
                     tz='CST',
                     max_export=0,
                     main_breaker_rating_A=2000, # disregard
                     n_phases=3,                               
                     theme='plotly_dark')

    data.import_data('data/bad_river/badriver_clinic_load_1min.csv', 
                     cols=['Datetime CST','Consumption kW'], 
                     rename=['Load'] )        
    
    #data.df = data.df.loc[:'2021']
    
    data.replaceNANsWithPersistence(lags_h=[1,2,24,25,168,48,72,96,120,144,336,504,672])
