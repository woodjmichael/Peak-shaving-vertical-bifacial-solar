__version__ = '14'

# %%
import sys
import pandas as pd
from bifacial_peak_shaving import *
from data_processing import *
pd.options.plotting.backend='plotly'

GPU = False
solar_scaler = 1.0

if len(sys.argv) > 1:
    for i,arg in enumerate(sys.argv[1:]):
        if arg == 'GPU':
            GPU = True
        if arg == '-s':
            solar_scaler = float(sys.argv[i+2])

# %%

output_filename_stub = f'Output/caltech_ev_mjw{__version__}_solar{solar_scaler:.1f}x_'

if GPU:
    data_dir = r'C:/Users/Admin/OneDrive - Politecnico di Milano/Data'
    output_filename_stub += 'GPU_'
else:
    data_dir = r'~/OneDrive/Data'

print('Output to ',output_filename_stub)

# %% [markdown]
# ## Load

# %%
load_file = r'/Load/Vehicle/ACN/old/acn_caltech.csv'

#shutil.copyfile(data_dir+load_file, r'./Data/'+load_file.split('/')[-1])

load = pd.read_csv(data_dir + r'/Load/Vehicle/ACN/old/acn_caltech.csv',
                   index_col=0,
                   parse_dates=True,
                   comment='#',)
                    #.resample('1h').mean()\
                    #.fillna(method='ffill')
                    #.loc['2018-4-26':'2019-3-22']
#load = pd.DataFrame({'Load (kW)':list(load.loc['2019-1-2 0:00':'2019-1-2 16:00','Load (kW)'])+list(load.loc[:,'Load (kW)'])},
#                     index=pd.date_range('2019-1-1 0:00',periods=8760,freq='1h'))
#load.loc['2019-5-20'] = load.loc['2019-5-22'].values
#load.loc['2019-5-21'] = load.loc['2019-5-22'].values
#load.index = pd.date_range('1901-1-1 0:00',periods=35040,freq='15min')
load = load.fillna(method='ffill')
#interval_min = int(load.index.to_series().diff().mean().seconds/60)
#load = load.resample('30min').mean()
load = load.loc['2018-5-1':'2019-2-28']
print(load.info())
print(load.describe())

# upsample
# load = load.resample('5min').interpolate(method='linear')
# load.loc[len(load)] = [pd.NA]
# load.index = list(load.index[:-1]) + [load.index[-2]+pd.Timedelta('5min')]
# load.loc[len(load)] = [pd.NA]
# load.index = list(load.index[:-1]) + [load.index[-2]+pd.Timedelta('5min')]
# load = load.fillna(method='ffill')

#plot_weekly(load['Load (kW)'])
#load.plot()

# %% [markdown]
# ## Solar
# 
# - 7 kWp, TMY
# 
# - Net zero capacity
# 
#   - $capacity[kWp] = production[kWh] \times yield [\frac{kWh}{kWp}]^{-1}$

# %%
solar_angles = ['s20','w90','s20w90_25_75','s20w90_50_50','s20w90_75_25']

dir_solar = data_dir + r'/Solar/Modelled/'

s20 = pd.concat((pd.read_csv(dir_solar+'pasadena_2018_15min_367mods_s20.csv',index_col=0,parse_dates=True,comment='#'),
                 pd.read_csv(dir_solar+'pasadena_2019_15min_367mods_s20.csv',index_col=0,parse_dates=True,comment='#')))

w90 = pd.concat((pd.read_csv(dir_solar+'pasadena_2018_15min_367mods_w90.csv',index_col=0,parse_dates=True,comment='#'),
                 pd.read_csv(dir_solar+'pasadena_2019_15min_367mods_w90.csv',index_col=0,parse_dates=True,comment='#')))

s20w90 = pd.concat((pd.read_csv(dir_solar+'pasadena_2018_15min_367mods_s20w90.csv',index_col=0,parse_dates=True,comment='#'),
                    pd.read_csv(dir_solar+'pasadena_2019_15min_367mods_s20w90.csv',index_col=0,parse_dates=True,comment='#')))


s20w90_25_75 = 0.25*s20 + 0.75*w90
s20w90_50_50 = 0.50*s20 + 0.50*w90
s20w90_75_25 = 0.75*s20 + 0.25*w90

solar = pd.concat((s20,w90,s20w90_25_75, s20w90_50_50, s20w90_75_25 ),axis=1)
solar.columns=solar_angles

#solar = solar.resample('1h').mean()



#net_zero_capacity = load['Load (kW)'].sum() / (solar.s20.sum()) # load=production / yield
#solar = solar * net_zero_capacity # scale to net 0


# upsample
# solar = solar.resample('5min').interpolate(method='linear')
# for _ in range(2):
#     solar.loc[len(solar)] = [pd.NA]*len(solar.columns)
#     solar.index = list(solar.index[:-1]) + [solar.index[-2]+pd.Timedelta('5min')]
# solar = solar.fillna(method='ffill')

solar = solar.loc[load.index] # make same length as load



# %% [markdown]
# Solar energy

# %%
solar.resample('1h').mean().sum()

# %% [markdown]
# Solar yield

# %%
solar.resample('1h').mean().sum()/(367*0.350)

solar = solar * solar_scaler

# %%
#for col in solar:
    #plot_daily(solar[col],title=col,interval_min=15)

# %% [markdown]
# ## Net load

# %%
solar.index = load.index # seems unnecessary
df = pd.concat((load,solar),axis=1)
df.columns = ['load'] + ['solar_'+x for x in solar.columns]
for angle in solar.columns:
    df[f'netload_{angle}'] = df['load'] - solar[angle]


# %% [markdown]
# # Baseline Costs
# 
# Load only, and net load given various solar

# %%
tou = TimeOfUseTariff(
   [{'months':[6,7,8,9],        'name':'All Hours',     'power_price':26.07,    'energy_price':0,       'hours':h((0,24))           },
    {'months':[6,7,8,9],        'name':'Off Peak',      'power_price':    0,    'energy_price':0.132,   'hours':h((0,14), (23,24)) },
    {'months':[6,7,8,9],        'name':'Partial Peak',  'power_price': 6.81,    'energy_price':0.159,   'hours':h((14,16),(21,23)) },
    {'months':[6,7,8,9],        'name':'Peak',          'power_price':32.90,    'energy_price':0.196,   'hours':h((16,21)) },
    {'months':[1,2,10,11,12],   'name':'All Hours',     'power_price':26.07,    'energy_price':0,       'hours':h((0,24)) },
    {'months':[1,2,10,11,12],   'name':'Off Peak',      'power_price':    0,    'energy_price':0.132,   'hours':h((0,14), (23,24)) },
    {'months':[1,2,10,11,12],   'name':'Partial Peak',  'power_price':    0,    'energy_price':     0,  'hours':h((14,16),(21,23)) },
    {'months':[1,2,10,11,12],   'name':'Peak',          'power_price': 2.20,    'energy_price':0.172,   'hours':h((16,21)) },
    {'months':[3,4,5],          'name':'All Hours',     'power_price':26.07,    'energy_price':0,       'hours':h((0,24)) },
    {'months':[3,4,5],          'name':'Super Off Peak','power_price':    0,    'energy_price':0.079,   'hours':h((9,14))},
    {'months':[3,4,5],          'name':'Off Peak',      'power_price':    0,    'energy_price':0.132,   'hours':h((0,9),  (14,16), (21,24)) },
    {'months':[3,4,5],          'name':'Peak',          'power_price': 2.20,    'energy_price':0.172,   'hours':h((16,21)) },                     ])


# %% [markdown]
# ## Minimize Cost
# 
# Grid search rough, fine, and gradient descent

# %%
angles = ['s20','s20w90_75_25','s20w90_50_50','s20w90_25_75','w90']
batt_kwhs = [25,50,75,100,125,150,200,400,600]

print('Angles',angles)
print('Batt_kwhs',batt_kwhs)

#angles = ['s20','w90']
#batt_kwhs = [25]

optimize_thresholds(df,tou,angles,batt_kwhs,output_filename_stub)

