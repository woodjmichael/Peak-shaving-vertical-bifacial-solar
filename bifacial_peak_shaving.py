# %%
''' Bifacial Peak Shaving
'''
__version__ = 24
import sys
import os
import shutil
import json
import pandas as pd
import numpy as np
import warnings
import yaml
from data_processing import calc_monthly_peaks, order_of_magnitude

warnings.simplefilter(action='ignore', category=FutureWarning)
# FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
# In a future version, this will no longer exclude empty or all-NA columns when determining the
# result dtypes. To retain the old behavior, exclude the relevant entries before the concat
# operation.

class dotdict(dict):
    '''dot.notation access to dictionary attributes'''

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_inputs(config_file:str=None) -> dotdict:
    
    if config_file is not None:
        #cfg = dotdict(json.load(open(config_file, 'r')))
                
        with open(config_file, 'r') as stream:
            d=yaml.safe_load(stream)
        cfg = dotdict(d)

    
    note = ''

    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg[:3] == '--f':
                break
            if arg[-4:] == 'json':
                config_file = arg
                cfg = dotdict(json.load(open(config_file, 'r')))
            if arg == '-a':
                cfg.solar_angles = [sys.argv[i + 2]]
                note += f'a{sys.argv[i + 2]}_'
            if arg == '-s':
                cfg.solar_scaler = float(sys.argv[i + 2])
                note += f's{cfg.solar_scaler:.1f}_'
            if arg == 'GPU':
                cfg.gpu = True
            if arg == 'TEST':
                cfg.test = True
                note += 'TEST_'             
            if arg == '-n':
                note += sys.argv[i + 2] + '_'     
                cfg = dotdict(json.load(open(config_file, 'r')))
                
    cfg['filename'] = config_file           
                
    if cfg.dev:
        note += 'DEV_'
    if cfg.gpu:
        cfg.data_dir = cfg.data_dir_gpu
        note += 'GPU_'
    if cfg.energy_price_sell_vector:
        cfg.energy_price_sell_vector = pd.read_csv( cfg.energy_price_sell_vector,
                                                    comment='#',
                                                    index_col=0,
                                                    parse_dates=True)
    if cfg.note is not None:
        note += cfg.note + '_'
    else:
        note = ''

    cfg.output_filename_stub = (
        f'Output/{config_file.split(".")[0]}_v{__version__}_s{cfg.solar_scaler:.1f}_{note}'
    )

    tou = {}
    for key in cfg.keys():
        if key[:3] == 'tou':
            tou[key] = cfg[key]
    cfg.tou = tou
    
    assert cfg.version == __version__, f'Version mismatch: {cfg.version} != {__version__}'                    

    return cfg


def h(*args) -> list:
    '''Create list of hours per day that defines a TOU period

    Args:
        *args (tuple of tuples): each tuple is (begin hour inclusive, end hour exclusive)

    Returns:
        list: hours of the TOU period
    '''
    hours = []
    for h_begin_inclusive, h_end_exclusive in args:
        hours += list(range(h_begin_inclusive, h_end_exclusive))
    return hours


class TimeOfUseTariff:
    def __init__(self, tou_raw):
        self.overlap = False  # if True some hours are in multiple periods
        if isinstance(tou_raw, str):
            tou_raw = json.load(open(tou_raw, 'r'))
        if isinstance(tou_raw, dict):
            self.periods = [dotdict(tou_raw[key]) for key in tou_raw.keys()]
        elif isinstance(tou_raw, list):
            self.periods = [dotdict(x) for x in tou_raw]
        if 'from_to_h' in self.periods[0].keys():
            self.create_hour_list()
        for period in self.periods:
            if period.power_price == 0:
                period['dead'] = True
            else:
                period['dead'] = False
        self.levels = len([x for x in self.periods if 1 in x['months']])
        self.check()
        print(f'Periods: {len(self.periods)} (overlap {self.overlap})')
        print('Levels:', self.levels)

    def __len__(
        self,
    ):  # small backwards compatability hack so len(tou) returns the number of levels
        return self.levels

    def create_hour_list(self):
        for period in self.periods:
            hours = []
            for subperiod in period['from_to_h']:
                begin, end = subperiod[0], subperiod[1]
                hours += h((begin, end))
            period['hours'] = hours

    def check(self):
        for month in range(1, 13):
            # has power and energy price (or is zero)
            for tou_month in self.get(month):
                assert tou_month['power_price'] is not None
                assert tou_month['energy_price_buy_sell'] is not None

            # each month 1-12 is represented and has the correct number of levels
            assert len([x for x in self.periods if month in x['months']]) == self.levels

            # each hour 0-23 is represented in 'hours'
            hours = []
            for tou_month in self.get(month):
                hours += tou_month['hours']

            if len(hours) > 24:
                self.overlap = True

            for hour in range(1, 24):
                assert hour in hours

        print('Tariff check: ok')

    def get(self, month):
        return [x for x in self.periods if month in x['months']]


def net_load_peak_reduction_variable_TOU(
    df: pd.DataFrame, angle1: str, angle2: str
) -> pd.DataFrame:
    '''Calculate the reduction of peak net load for a range of TOU windows

    MAY NOT WORK CORRECTLY

    Args:
        df (pd.DataFrame): net load data
        angle1 (str): reference angle (usually s20)
        angle2 (str): comparison angle

    Returns:
        pd.DataFrame: results
    '''
    all_results = pd.DataFrame(
        columns=[
            'TOU begin [h]',
            'TOU end [h]',
            'max peak reduction [kW]',
            'avg peak reduction [kW]',
            'min peak reduction [kW]',
        ]
    )
    for peak_begin in list(range(8, 21)):
        for peak_end in list(range(9, 22)):
            if (peak_end - peak_begin) >= 2:
                peaks = calc_monthly_peaks(
                    df[f'netload_{angle1}'], peak_begin, peak_end
                )

                peaks = pd.concat(
                    (
                        peaks,
                        calc_monthly_peaks(
                            df[f'netload_{angle2}'], peak_begin, peak_end
                        ),
                    ),
                    axis=1,
                )
                peaks['peak_reduction'] = (
                    peaks[f'netload_{angle1} kw'] - peaks[f'netload_{angle2} kw']
                )

                pr_max = round(peaks.peak_reduction.max(), 1)
                pr_avg = round(peaks.peak_reduction.mean(), 1)
                pr_min = round(peaks.peak_reduction.min(), 1)
                # print(f'{peak_begin:3d} {peak_end:3d} -- {pr_max:7.2f} {pr_avg:7.2f} {pr_min:7.2f}')
                all_results.loc[len(all_results)] = [
                    peak_begin,
                    peak_end,
                    pr_max,
                    pr_avg,
                    pr_min,
                ]
                # results

    all_results['TOU begin [h]'] = all_results['TOU begin [h]'].astype(int)
    all_results['TOU end [h]'] = all_results['TOU end [h]'].astype(int)
    return all_results


def update_soe(soe: list, power: float, soe0: float, interval_min: int = 60) -> float:
    '''Update state of energy from actual battery power

    Args:
        soe (list of floats): state of energy vector
        power (float): battery power (+ is discharge, - is charge)
        soe0 (float): initial state of energy

    Returns:
        float: next timestep state of energy
    '''
    if len(soe) == 0:
        last_soe = soe0
    else:
        last_soe = soe[-1]

    energy = power * (
        interval_min / 60
    )  # kwh = kw * interval_h = kw * interval_min / (60 min/h)
    # next_soe = last_soe - energy*0.9

    if power > 0:  # discharge
        next_soe = last_soe - energy / 0.9
    elif power < 0:  # charge
        next_soe = last_soe - energy * 0.9
    else:
        next_soe = last_soe

    return next_soe


def batt_request(
    soe: list, power: float, soe0: float, soe_max: float, interval_min: int = 60
) -> float:
    '''Validate battery power based on state of energy max and min

    Args:
        soe (list of floats): state of energy vector
        power (float): battery power (+ is discharge, - is charge)
        soe0 (float): initial state of energy
        soe_max (float): maximum state of energy

    Returns:
        float: valdiated battery power (+ is discharge, - is charge)
    '''

    if len(soe) == 0:
        last_soe = soe0
    else:
        last_soe = soe[-1]

    if power > 0:  # discharge
        available_energy = (last_soe - 0) * 0.9
        available_power = available_energy / (interval_min / 60)
        power = min(power, available_power)
    elif power < 0:  # charge
        available_energy = (soe_max - last_soe) / 0.9
        available_power = available_energy / (interval_min / 60)
        power = max(power, -1 * available_power)
    else:
        power = 0
    return power


def peak_shaving_sim_old(df2, nl_col, threshold, soe0, TOU_hours, soe_max=None):
    if soe_max is None:
        soe_max = soe0
    tol = 0.001  # tolerance
    df2 = df2.copy(deep=True)
    tou_first, tou_last = TOU_hours[0], TOU_hours[-1]
    batt, soe = [], []
    for net_load, t in zip(df2[nl_col], df2.index):
        if t.hour in TOU_hours:
            batt.append(batt_request(soe, net_load - threshold, soe0, soe_max))
        else:
            batt.append(min(0, batt_request(soe, net_load - threshold, soe0, soe_max)))
        soe.append(update_soe(soe, batt[-1], soe0))

    df2['batt'] = batt
    df2['soe'] = soe

    df2['utility'] = df2[nl_col] - df2.batt

    return (
        any(
            df2[[tou_first <= h <= tou_last for h in df2.index.hour]].utility
            > (threshold + tol)
        ),
        df2,
    )


def peak_shaving_sim(
    df: pd.DataFrame,
    netload_col: str,
    soe_max: float,
    thresholds: list = None,
    TOU: list = None,
    soe0_pu: float = 1,
    soc=True,
    utility_chg_max: float = 0,
):
    '''Simulate peak shaving battery dispatch

    Args:
        df (pd.DataFrame): load solar and netload data
        netload_col (str): netload column name
        soe_max (float): maximum battery state of energy
        thresholds (float): battery will be dispatched to keep import from utility below this value
            during threshold_h
        TOU (list of dicts): list of TOU dicts like {'price':float and 'hours':list of ints}
        soe0 (float): initial battery state of energy
        utility_chg_max (float, optional): max absolute value that battery will charge at during
            non-TOU hours. Defaults to 0.

    Returns:
        (bool,pd.DataFrame): sim failed for this battery size?, dispatch vectors
    '''

    df = df.copy(deep=True)  # we'll be modifying this
    interval_min = int(df.index.to_series().diff().mean().seconds / 60)

    # TOU
    if len(TOU) >= 1:
        threshold0_h = TOU[0]['hours']
    if len(TOU) >= 2:
        threshold1_h = TOU[1]['hours']
    if len(TOU) >= 3:
        threshold2_h = TOU[2]['hours']
    threshold_all_h = []
    for tou_level in TOU:
        threshold_all_h += tou_level['hours']

    # thresholds
    if len(thresholds) >= 1:
        threshold0 = thresholds[0]
    if len(thresholds) >= 2:
        threshold1 = thresholds[1]
    if len(thresholds) >= 3:
        threshold2 = thresholds[2]

    soe0 = soe0_pu * soe_max
    tol = 0.001  # tolerance

    batt, soe = [], []
    for netload, t in zip(df[netload_col], df[netload_col].index):
        if t.hour in threshold_all_h:
            threshold = 1e3
            if t.hour in threshold0_h:
                threshold = min(threshold, threshold0)
            if t.hour in threshold1_h:
                threshold = min(threshold, threshold1)
            if t.hour in threshold2_h:
                threshold = min(threshold, threshold2)
            batt.append(
                batt_request(soe, netload - threshold, soe0, soe_max, interval_min)
            )
        else:
            batt.append(
                min(
                    0,
                    batt_request(
                        soe, min(netload, -utility_chg_max), soe0, soe_max, interval_min
                    ),
                )
            )
        soe.append(update_soe(soe, batt[-1], soe0, interval_min))

    df['batt'] = batt
    df['soe'] = soe
    df['utility'] = df[netload_col] - df.batt
    if soc:
        df['soc'] = df.soe / soe_max

    # determine if there was a failure
    h_0, h_f = threshold0_h[0], threshold0_h[-1]
    failure = any(
        df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold0 + tol)
    )
    if threshold1 is not None:
        h_0, h_f = threshold1_h[0], threshold1_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold1 + tol)
        )
    if threshold2 is not None:
        h_0, h_f = threshold2_h[0], threshold2_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold2 + tol)
        )

    return failure, df


def peak_shaving_sim_4tou(
    df: pd.DataFrame,
    netload_col: str,
    soe_max: float,
    thresholds: list = None,
    tou=None,
    soe0_pu: float = 1,
    soc=True,
    utility_chg_max: float = 0,
    return_cost = False
):
    '''Simulate peak shaving battery dispatch

    Args:
        df (pd.DataFrame): load solar and netload data
        netload_col (str): netload column name
        soe_max (float): maximum battery state of energy
        thresholds (float): battery will be dispatched to keep import from utility below this value
            during threshold_h
        tou (list of dicts): list of TOU dicts like {'price':float and 'hours':list of ints}
        soe0 (float): initial battery state of energy
        utility_chg_max (float, optional): max absolute value that battery will charge at during
            non-TOU hours. Defaults to 0.

    Returns:
        (bool,pd.DataFrame): sim failed for this battery size?, dispatch vectors
    '''

    if isinstance(tou, TimeOfUseTariff):
        month = df.index.month.unique()[0]
        tou = tou.get(month)

    df = df.copy(deep=True)  # we'll be modifying this
    interval_min = int(df.index.to_series().diff().mean().seconds / 60)

    # tou
    if len(tou) >= 1:
        threshold0_h = tou[0]['hours']
    if len(tou) >= 2:
        threshold1_h = tou[1]['hours']
    if len(tou) >= 3:
        threshold2_h = tou[2]['hours']
    if len(tou) >= 4:
        threshold3_h = tou[3]['hours']
    threshold_all_h = []
    for tou_level in tou:
        threshold_all_h += tou_level['hours']

    # thresholds
    if len(thresholds) >= 1:
        threshold0 = thresholds[0]
    if len(thresholds) >= 2:
        threshold1 = thresholds[1]
    if len(thresholds) >= 3:
        threshold2 = thresholds[2]
    if len(thresholds) >= 4:
        threshold3 = thresholds[3]

    soe0 = soe0_pu * soe_max
    tol = 0.001  # tolerance

    batt, soe = [], []
    for netload, t in zip(df[netload_col], df[netload_col].index):
        if t.hour in threshold_all_h:
            threshold = 1e3
            if t.hour in threshold0_h:
                threshold = min(threshold, threshold0)
            if t.hour in threshold1_h:
                threshold = min(threshold, threshold1)
            if t.hour in threshold2_h:
                threshold = min(threshold, threshold2)
            if t.hour in threshold3_h:
                threshold = min(threshold, threshold3)
            batt.append(
                batt_request(soe, netload - threshold, soe0, soe_max, interval_min)
            )
        else:
            batt.append(
                min(
                    0,
                    batt_request(
                        soe, min(netload, -utility_chg_max), soe0, soe_max, interval_min
                    ),
                )
            )
        soe.append(update_soe(soe, batt[-1], soe0, interval_min))

    df['batt'] = batt
    df['soe'] = soe
    df['utility'] = df[netload_col] - df.batt
    if soc:
        df['soc'] = df.soe / soe_max

    # determine if there was a failure
    h_0, h_f = threshold0_h[0], threshold0_h[-1]
    failure = any(
        df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold0 + tol)
    )
    if threshold1 is not None:
        h_0, h_f = threshold1_h[0], threshold1_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold1 + tol)
        )
    if threshold2 is not None:
        h_0, h_f = threshold2_h[0], threshold2_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold2 + tol)
        )
    if threshold3 is not None:
        h_0, h_f = threshold3_h[0], threshold3_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold3 + tol)
        )

    if return_cost:
        return failure, df, calc_cost(df.utility, tou)
    else:
        return failure, df

def peak_shaving_sim_Xtou(
    df: pd.DataFrame,
    netload_col: str,
    soe_max: float,
    thresholds: list = None,
    tou=None,
    soe0_pu: float = 1,
    soc=True,
    utility_chg_max: float = 0,
):
    '''Simulate peak shaving battery dispatch

    Args:
        df (pd.DataFrame): load solar and netload data
        netload_col (str): netload column name
        soe_max (float): maximum battery state of energy
        thresholds (float): battery will be dispatched to keep import from utility below this value
            during threshold_h
        tou (list of dicts): list of TOU dicts like {'price':float and 'hours':list of ints}
        soe0 (float): initial battery state of energy
        utility_chg_max (float, optional): max absolute value that battery will charge at during
            non-TOU hours. Defaults to 0.

    Returns:
        (bool,pd.DataFrame): sim failed for this battery size?, dispatch vectors
    '''
    
    threshold0,threshold1,threshold2,threshold3 = None,None,None,None

    if isinstance(tou, TimeOfUseTariff):
        month = df.index.month.unique()[0]
        tou = tou.get(month)

    df = df.copy(deep=True)  # we'll be modifying this
    interval_min = int(df.index.to_series().diff().mean().seconds / 60)

    # tou
    if len(tou) >= 1:
        threshold0_h = tou[0]['hours']
    if len(tou) >= 2:
        threshold1_h = tou[1]['hours']
    if len(tou) >= 3:
        threshold2_h = tou[2]['hours']
    if len(tou) >= 4:
        threshold3_h = tou[3]['hours']
    threshold_all_h = []
    for tou_level in tou:
        threshold_all_h += tou_level['hours']

    # thresholds
    if len(thresholds) >= 1:
        threshold0 = thresholds[0]
    if len(thresholds) >= 2:
        threshold1 = thresholds[1]
    if len(thresholds) >= 3:
        threshold2 = thresholds[2]
    if len(thresholds) >= 4:
        threshold3 = thresholds[3]

    soe0 = soe0_pu * soe_max
    tol = 0.001  # tolerance

    batt, soe = [], []
    for netload, t in zip(df[netload_col], df[netload_col].index):
        if t.hour in threshold_all_h:
            threshold = 1e3
            if len(tou) >= 1:
                if t.hour in threshold0_h:
                    threshold = min(threshold, threshold0)
            if len(tou) >= 2:
                if t.hour in threshold1_h:
                    threshold = min(threshold, threshold1)
            if len(tou) >= 3:
                if t.hour in threshold2_h:
                    threshold = min(threshold, threshold2)
            if len (tou) >= 4:
                if t.hour in threshold3_h:
                    threshold = min(threshold, threshold3)
            batt.append(
                batt_request(soe, netload - threshold, soe0, soe_max, interval_min)
            )
        else:
            batt.append(
                min(
                    0,
                    batt_request(
                        soe, min(netload, -utility_chg_max), soe0, soe_max, interval_min
                    ),
                )
            )
        soe.append(update_soe(soe, batt[-1], soe0, interval_min))

    df['batt'] = batt
    df['soe'] = soe
    df['utility'] = df[netload_col] - df.batt
    if soc:
        df['soc'] = df.soe / soe_max

    # determine if there was a failure
    h_0, h_f = threshold0_h[0], threshold0_h[-1]
    failure = any(
        df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold0 + tol)
    )
    if threshold1 is not None:
        h_0, h_f = threshold1_h[0], threshold1_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold1 + tol)
        )
    if threshold2 is not None:
        h_0, h_f = threshold2_h[0], threshold2_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold2 + tol)
        )
    if threshold3 is not None:
        h_0, h_f = threshold3_h[0], threshold3_h[-1]
        failure = failure or any(
            df[[h_0 <= h <= h_f for h in df.index.hour]].utility > (threshold3 + tol)
        )

    return failure, df


def find_smallest_soe0(
    df2,
    net_load_col,
    threshold,
    batt_kwh_max,
    TOU_hours,
    step=1,
    output=True,
    utility_max_chg_pu=None,
    utility_max_chg=100000,
    soe0_pu=1,
):
    if len(threshold) > 3:
        print('Not set up for 4 TOUs')
        sys.exit()

    # Find a good initial batter kwh capacity by scaling the passed value
    for multiplier in [1, 1.1, 2, 5, 10, 25, 100]:
        batt_kwh = batt_kwh_max * multiplier
        fail, _ = peak_shaving_sim(
            df2,
            net_load_col,
            threshold,
            batt_kwh,
            TOU_hours,
            soe0_pu,
            min(utility_max_chg_pu * batt_kwh, utility_max_chg),
        )
        if not fail:
            break
        else:
            print(f'No solution found for battery max kWh multiplier = {multiplier}')
            step = step * 10 ** order_of_magnitude(
                multiplier
            )  # increase step with order of magnitude of multiplier

    # Find the lowest batt_kwh without peak shaving failures
    if fail == False:
        for kwh in [x * step for x in range(int(batt_kwh / step), 0, -1)]:
            fail, _ = peak_shaving_sim(
                df2,
                net_load_col,
                threshold,
                kwh,
                TOU_hours,
                soe0_pu,
                min(utility_max_chg_pu * kwh, utility_max_chg),
            )
            if fail == True:
                break
    else:
        print('Get out of here with your AAA batteries!!')
        return 0, df2

    # Solution is the smallest size that didn't fail
    kwh += step
    fail, df3 = peak_shaving_sim(
        df2,
        net_load_col,
        threshold,
        kwh,
        TOU_hours,
        soe0_pu,
        min(utility_max_chg_pu * kwh, utility_max_chg),
    )
    if (fail == False) and output:
        print(
            f'Solution found: {net_load_col}, Threshold kw = {threshold:.1f}, Battery kWh = {kwh:.1f},'
        )
    else:
        print(
            f'Failure to find solution: {net_load_col}, Threshold kw = {threshold:.1f}, Battery kWh max = {batt_kwh_max:.1f},'
        )

    return kwh, df3


def calc_power_cost(ds: pd.Series, tou: list, peak_interval_min: int = 60) -> float:
    ds = ds.resample(f'{peak_interval_min}min').mean()
    cost = 0
    for tou_level in tou:
        price, hours = tou_level['price'], tou_level['hours']
        max_power = ds[[True if h in hours else False for h in ds.index.hour]].max()
        cost += max(0, max_power) * price
    return cost


def calc_cost(ds: pd.Series, tou, peak_interval_min: int = 60) -> float:
    cost = 0
    ds = ds.resample(f'{peak_interval_min}min').mean()
    #ds[ds < 0] = 0
    if isinstance(tou, list):
        for tou_level in tou:
            power_price = tou_level['power_price']
            energy_price = tou_level['energy_price']
            hours = tou_level['hours']
            max_power = ds[[True if h in hours else False for h in ds.index.hour]].max()
            energy = (
                ds[[True if h in hours else False for h in ds.index.hour]]
                .resample('1h')
                .mean()
                .sum()
            )
            cost += max(0, max_power) * power_price + max(0, energy) * energy_price
    elif isinstance(tou, TimeOfUseTariff):
        for year in ds.index.year.unique():
            ds_year = ds[ds.index.year == year]
            for month in ds_year.index.month.unique():
                ds_month = ds_year[ds_year.index.month == month]
                for tou_level in tou.get(month):
                    power_price = tou_level.power_price
                    energy_price_buy = tou_level.energy_price_buy_sell[0]
                    energy_price_sell = tou_level.energy_price_buy_sell[1]
                    hours = tou_level.hours
                    ds_month_tou = ds_month[[True if h in hours else False for h in ds_month.index.hour]]
                    max_power = ds_month_tou.max()
                    ds_month_tou = ds_month_tou.resample('1h').mean()
                    energy_pos = ds_month_tou[[True if x>0 else False for x in ds_month_tou.values]].sum()
                    energy_neg = ds_month_tou[[True if x<0 else False for x in ds_month_tou.values]].sum()
                    
                    cost += (max(0, max_power) * power_price \
                            + max(0, energy_pos) * energy_price_buy \
                            + min(0, energy_neg) * energy_price_sell)
                    
                    if cfg.energy_price_sell_vector is not False:
                        ds_month = ds_month[ds_month.values<0]
                        sell_prices = cfg.energy_price_sell_vector.loc[ds_month.index,:]
                        cost += (ds_month.values * sell_prices.Price.values).sum()
    return cost


def f(thresholds, df_month, angle, batt_kwh, tou):
    th0, th1, th2 = thresholds
    fail, dispatch = peak_shaving_sim(
        df_month,
        f'netload_{angle}',
        batt_kwh,
        [th0, th1, th2],
        tou,
        utility_chg_max=batt_kwh,
    )
    cost = calc_cost(dispatch.utility, tou)
    return cost, fail, dispatch


def f4t(thresholds, df_month, angle, batt_kwh, tou):
    th0, th1, th2, th3 = thresholds
    fail, dispatch = peak_shaving_sim_4tou(
        df_month,
        f'netload_{angle}',
        batt_kwh,
        [th0, th1, th2, th3],
        tou,
        utility_chg_max=batt_kwh,
    )
    cost = calc_cost(dispatch.utility, tou)
    return cost, fail, dispatch


def grad_f(th_i, df_month, angle, batt_kwh, tou):
    # c,fail,_ = f(th_i)

    d = 0.05

    c0_0, fail, _ = f(
        [x + dx for x, dx in zip(th_i, [-d, 0, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c0_0, fail, _ = f(th_i, df_month, angle, batt_kwh, tou)
    c0_1, fail, _ = f(
        [x + dx for x, dx in zip(th_i, [d, 0, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c0_1, fail, _ = f(th_i, df_month, angle, batt_kwh, tou)

    c1_0, fail, _ = f(
        [x + dx for x, dx in zip(th_i, [0, -d, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c1_0, fail, _ = f(th_i, df_month, angle, batt_kwh, tou)
    c1_1, fail, _ = f(
        [x + dx for x, dx in zip(th_i, [0, d, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c1_1, fail, _ = f(th_i, df_month, angle, batt_kwh, tou)

    c2_0, fail, _ = f(
        [x + dx for x, dx in zip(th_i, [0, 0, -d])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c2_0, fail, _ = f(th_i, df_month, angle, batt_kwh, tou)
    c2_1, fail, _ = f(
        [x + dx for x, dx in zip(th_i, [0, 0, d])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c2_1, fail, _ = f(th_i, df_month, angle, batt_kwh, tou)

    return (c0_0 - c0_1, c1_0 - c1_1, c2_0 - c2_1)


def grad_f4t(th_i, df_month, angle, batt_kwh, tou):
    # c,fail,_ = f(th_i)

    d = 0.05

    c0_0, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [-d, 0, 0, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c0_0, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)
    c0_1, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [d, 0, 0, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c0_1, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)

    c1_0, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [0, -d, 0, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c1_0, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)
    c1_1, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [0, d, 0, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c1_1, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)

    c2_0, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [0, 0, -d, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c2_0, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)
    c2_1, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [0, 0, d, 0])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c2_1, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)

    c3_0, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [0, 0, 0, -d])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c3_0, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)
    c3_1, fail, _ = f4t(
        [x + dx for x, dx in zip(th_i, [0, 0, 0, d])], df_month, angle, batt_kwh, tou
    )
    if fail == True:
        c3_1, fail, _ = f4t(th_i, df_month, angle, batt_kwh, tou)

    return (c0_0 - c0_1, c1_0 - c1_1, c2_0 - c2_1, c3_0 - c3_1)


def process_output(filename=None,df=None):
    if filename is not None:
        output = pd.read_csv(filename, index_col=0)
    if df is not None:
        output = df.copy(deep=True)
        
    output = output[['angle', 'batt kwh', 'total cost']]
    angles = list(output.angle.unique())
    batt_sizes = output['batt kwh'].unique()
    results = pd.DataFrame([], index=batt_sizes)

    # cost
    for angle in angles:
        results[angle] = output[output.angle == angle]['total cost'].values.round(0)

    angles.remove('s20')

    # cost reduction
    for angle in angles:
        results[f'{angle} red'] = (results.s20 - results[angle]).round(0)

    # cost reduction percent
    for angle in angles:
        results[f'{angle} red%'] = (
            100 * (results.s20 - results[angle]) / results.s20
        ).round(2)

    return results


def read_load_data(cfg):
    load = pd.read_csv(
        cfg.data_dir + cfg.load_file,
        index_col=0,
        parse_dates=True,
        comment='#',
    )
    load = load.ffill()
    load = load.loc[cfg.sim_begin:cfg.sim_end]
    return load


def read_and_scale_solar(cfg, load_index):
    dir_solar = cfg.data_dir + cfg.solar_files
    s20 = pd.concat(
        (
            pd.read_csv(
                dir_solar + 'pasadena_2018_15min_367mods_s20.csv',
                index_col=0,
                parse_dates=True,
                comment='#',
            ),
            pd.read_csv(
                dir_solar + 'pasadena_2019_15min_367mods_s20.csv',
                index_col=0,
                parse_dates=True,
                comment='#',
            ),
        )
    )

    w90 = pd.concat(
        (
            pd.read_csv(
                dir_solar + 'pasadena_2018_15min_367mods_w90.csv',
                index_col=0,
                parse_dates=True,
                comment='#',
            ),
            pd.read_csv(
                dir_solar + 'pasadena_2019_15min_367mods_w90.csv',
                index_col=0,
                parse_dates=True,
                comment='#',
            ),
        )
    )

    # s20w90 = pd.concat(
    #     (
    #         pd.read_csv(
    #             dir_solar + 'pasadena_2018_15min_367mods_s20w90.csv',
    #             index_col=0,
    #             parse_dates=True,
    #             comment='#',
    #         ),
    #         pd.read_csv(
    #             dir_solar + 'pasadena_2019_15min_367mods_s20w90.csv',
    #             index_col=0,
    #             parse_dates=True,
    #             comment='#',
    #         ),
    #     )
    # )

    solar = pd.concat((s20, w90), axis=1)
    solar.columns = ['s20', 'w90']

    combination_angles = cfg.solar_angles.copy()
    if 's20' in combination_angles:
        combination_angles.remove('s20')
    if 'w90' in combination_angles:
        combination_angles.remove('w90')

    for angle in combination_angles:
        s20_frac = float(angle.split('_')[1]) / 100
        w90_frac = float(angle.split('_')[2]) / 100
        combination = s20_frac * s20 + w90_frac * w90
        solar.insert(len(solar.columns) - 1, angle, combination)

    solar = solar.loc[load_index]  # make same length as load
    solar = solar * cfg.solar_scaler
    return solar


def calculate_net_load(load, solar):
    solar.index = load.index  # seems unnecessary
    df = pd.concat((load, solar), axis=1)
    df.columns = ['load'] + ['solar_' + x for x in solar.columns]
    for angle in solar.columns:
        df[f'netload_{angle}'] = df['load'] - solar[angle]
    return df


def optimize_thresholds(
    cfg,
    df,
    tou,
    test=False,
    tou1_is_tou0=True,
    tou2_is_tou0=True,
    process_results=True,
):
    '''Grid search and then gradient descent optimization of peak shaving thresholds. Option to
    not optimize the second (tou1) TOU window, but instead to hold it equal to the first (tou0).

    Args:
        df (pd.DataFrame): data
        tou (dict): TOU tariff definition
        angles (list): list of angles to optimize (e.g. s20, s20w90_50_50, w90)
        batt_kwhs (list): list of angles to optimize (e.g. 25,50,75,100,125)
        output_filename_stub (str, optional): beginning of path to output file (will be appended
            with datetime code). Defaults to 'Output/bifacil_peak_shaving_'.
        test (bool, optional): only do the first angle-battery-month combination to test code.
            Defaults to False.
        tou1_is_tou0 (bool, optional): don't optimize the second (tou1) window, make value equal to
            first window (tou0). Defaults to True.
    '''
    angles, batt_kwhs, output_filename_stub, test = (
        cfg.solar_angles,
        cfg.batt_kwhs,
        cfg.output_filename_stub,
        cfg.test
    )
    
    tic = pd.Timestamp.now()
    
    outdir = output_filename_stub + tic.strftime('%y%m%d_%H%M') + '/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        outdir = outdir[:-1]+tic.strftime('%S') + '/'
        os.mkdir(outdir)

    year_months = [
        f'{y}-{m}'
        for y in df.index.year.unique()
        for m in df.loc[str(y)].index.month.unique()
    ]

    search_max = cfg.grid_search_max
    step = cfg.grid_search_step

    runs = []
    for batt_kwh in batt_kwhs:
        for angle in angles:
            if cfg.grid_search_prices:
                for p_p in cfg.power_prices:
                    for p_s in cfg.sell_prices:
                        for p_e2 in [x+p_s for x in cfg.energy_prices_adder_2]:
                            for p_e3 in [x+p_e2 for x in cfg.energy_prices_adder_3]:
                                runs.append((angle, batt_kwh, p_p, p_s, p_e2, p_e3))
            else:
                runs.append((angle, batt_kwh))
            
    if test == True:
        runs = runs[:1]
        year_months = year_months[:1]
    if cfg.grid_search_prices:
        output = pd.DataFrame([], columns=['angle', 'batt kwh', 'Pp','Ps','Pe2','Pe3', 'total cost'] + year_months)
    else:
        output = pd.DataFrame([], columns=['angle', 'batt kwh', 'total cost'] + year_months)
    
    for run in runs:
        if cfg.grid_search_prices:
            angle, batt_kwh, p_p, p_s, p_e2, p_e3 = run
            tou.periods[3].power_price = p_p
            tou.periods[3].energy_price_buy_sell = [p_e3, p_s]
            tou.periods[2].energy_price_buy_sell = [p_e2, p_s]
        else:
            angle, batt_kwh = run
        
        best_monthly = pd.DataFrame(
            [],
            columns=[
                'year-month',
                'threshold0',
                'threshold1',
                'threshold2',
                'threshold3',
                'cost',
            ],
        )
        for year_month in year_months:
            _df = df.loc[year_month][['load', f'solar_{angle}', f'netload_{angle}']]
            t_newmonth = pd.Timestamp.now()
            print(
                f'\n{t_newmonth} Angle {angle} batt_kwh {batt_kwh } begin new month {year_month}'
            )

            r = pd.DataFrame(
                [],
                columns=[
                    'i',
                    'fail',
                    'th0',
                    'th1',
                    'th2',
                    'th3',
                    'c',
                    'deltac',
                    'dc',
                ],
            )
            # r = pd.DataFrame()
            #r_cols = ['i', 'fail', 'th0', 'th1', 'th2', 'th3', 'c', 'deltac', 'dc']

            #
            # Rough grid search
            #

            k = 0
            
            th0_range = range(0, search_max, step)
            if tou.periods[0].dead:
                th0_range = [1e3]
            th1_range = range(0, search_max, step)
            if tou.periods[1].dead:
                th1_range = [1e3]
            th2_range = range(0, search_max, step)
            if tou.periods[2].dead:
                th2_range = [1e3]
            th3_range = range(0, search_max, step)
            if tou.periods[3].dead:
                th3_range = [1e3]
            
            for th0 in th0_range:
                if tou1_is_tou0:
                    th1_range = [th0]
                for th1 in th1_range:
                    if tou2_is_tou0:
                        th2_range = [th0]
                    th1 = min(th0, th1)  # remove if T0 is not 'all hours'
                    for th2 in th2_range:
                        # remove if T0 is not 'all hours'
                        th2 = min(th0, th2)
                        for th3 in th3_range:
                            # remove if T0 is not 'all hours'
                            th3 = min(th0, th3)
                            k += 1
                            fail, dispatch = peak_shaving_sim_4tou(
                                _df,
                                f'netload_{angle}',
                                batt_kwh,
                                [th0, th1, th2, th3],
                                tou,
                                utility_chg_max=batt_kwh,
                            )
                            # if fail == False:
                            cost = calc_cost(dispatch.utility, tou)
                            r.loc[len(r)] = [
                                k,
                                fail,
                                th0,
                                th1,
                                th2,
                                th3,
                                cost,
                                pd.NA,
                                pd.NA,
                            ]
                            # r = pd.concat(
                            #     [
                            #         r if not r.empty else None,
                            #         pd.DataFrame(
                            #             {
                            #                 'i': k,
                            #                 'fail': fail,
                            #                 'th0': th0,
                            #                 'th1': th1,
                            #                 'th2': th2,
                            #                 'th3': th3,
                            #                 'c': cost,
                            #                 'deltac': pd.NA,
                            #                 'dc': pd.NA,
                            #             },
                            #             index=[k],
                            #         ),
                            #     ]
                            # )

            if len(r[r.fail == False]) > 0:
                best_cost = r[r.fail == False]['c'].min()
                imin = r[r.fail == False]['c'].idxmin()
                best_th0 = r.loc[imin].th0
                best_th1 = r.loc[imin].th1
                best_th2 = r.loc[imin].th2
                best_th3 = r.loc[imin].th3
            else:
                print(
                    f'\n\n\n /// No viable solutions for angle {angle} batt_kwh {batt_kwh} year-month {year_month} search_max {search_max} /// \n\n\n'
                )
                sys.exit()

            t_rgs = pd.Timestamp.now()
            dt = (t_rgs - t_newmonth).seconds
            print(
                f'{t_rgs} (+{dt}s) Rough grid search done for {year_month} thresholds=({best_th0:.1f},{best_th1:.1f},{best_th2:.1f},{best_th3:.1f}), cost={best_cost:.1f} '
            )

            #
            # Fine grid search
            #
            
            th0_range = range(
                max(0, int(best_th0 - step)),
                int(best_th0 + step),
                cfg.grid_search_fine_step,)
            if tou.periods[0].dead:
                th0_range = [1e3]
            th1_range = range(
                        max(0, int(best_th1 - step)),
                        int(best_th1 + step),
                        cfg.grid_search_fine_step,)
            if tou.periods[1].dead:
                th1_range = [1e3]
            th2_range = range(
                        max(0, int(best_th2 - step)),
                        int(best_th2 + step),)
            if tou.periods[2].dead:
                th2_range = [1e3]
            th3_range = range(
                            max(0, int(best_th3 - step)),
                            int(best_th3 + step),)
            if tou.periods[3].dead:
                th3_range = [1e3]                

            for th0 in th0_range:
                if tou1_is_tou0:
                    th1_range = [th0]
                for th1 in th1_range:
                    th1 = min(th0, th1)
                    if tou2_is_tou0:
                        th2_range = [th0]                    
                    for th2 in th2_range:
                        th2 = min(th0, th2)
                        for th3 in th3_range:
                            th3 = min(th0, th3)
                            k += 1
                            fail, dispatch = peak_shaving_sim_4tou(
                                _df,
                                f'netload_{angle}',
                                batt_kwh,
                                [th0, th1, th2, th3],
                                tou,
                                utility_chg_max=batt_kwh,
                            )
                            cost = calc_cost(dispatch.utility, tou)
                            r.loc[len(r)] = [
                                k,
                                fail,
                                th0,
                                th1,
                                th2,
                                th3,
                                cost,
                                pd.NA,
                                pd.NA,
                            ]

            best_cost = r[r.fail == False]['c'].min()
            imin = r[r.fail == False]['c'].idxmin()
            best_th0 = r.loc[imin].th0
            best_th1 = r.loc[imin].th1
            best_th2 = r.loc[imin].th2
            best_th3 = r.loc[imin].th3

            t_fgs = pd.Timestamp.now()
            dt = (t_fgs - t_rgs).seconds
            print(
                f'{t_fgs} (+{dt}s) Fine grid search done for {year_month} thresholds=({best_th0:.1f},{best_th1:.1f},{best_th2:.1f},{best_th3:.1f}), cost={best_cost:.1f} '
            )

            #
            # Gradient descent
            #
            
            if cfg.gradient_descent:
                
                LR = (0.1, 0.1, 0.1, 0.1)

                c = []
                th = [[best_th0, best_th1, best_th2, best_th3]]
                cost, fail, dispatch = f4t(th[0], _df, angle, batt_kwh, tou)
                c.append(cost)

                # final_countdown = 20
                cmin = cost
                patience_counter = 0
                for i in range(1, 500):
                    dc = grad_f4t(th[i - 1], _df, angle, batt_kwh, tou)  # dy
                    dth = [
                        (dx + nz) * lr for dx, lr, nz in zip(dc, LR, np.random.rand(4))
                    ]  # dx
                    new_th = [
                        max(0, x + dx) for x, dx in zip(th[i - 1], dth)
                    ]  # x2 = x1 + dx
                    new_th[1] = min(new_th[0], new_th[1])  # all th1 <= th0
                    new_th[2] = min(new_th[0], new_th[2])  # all th2 <= th0
                    new_th[3] = min(new_th[0], new_th[3])  # all th3 <= th0

                    if tou.periods[0].dead:
                        new_th[0] = 1e3
                    if tou.periods[1].dead:
                        new_th[1] = 1e3
                    if tou1_is_tou0:
                        new_th[1] = new_th[0]
                    if tou.periods[2].dead:
                        new_th[2] = 1e3
                    if tou.periods[3].dead:
                        new_th[3] = 1e3

                    th.append(new_th)
                    cost, fail, dispatch = f4t(th[i], _df, angle, batt_kwh, tou)
                    c.append(cost)

                    # r = pd.concat(
                    #     [
                    #         r if not r.empty else None,
                    #         pd.DataFrame(
                    #             {
                    #                 'i': i + k,
                    #                 'fail': fail,
                    #                 'th0': round(th[i][0], 3),
                    #                 'th1': round(th[i][1], 3),
                    #                 'th2': round(th[i][2], 3),
                    #                 'th3': round(th[i][3], 3),
                    #                 'c': c[i],
                    #                 'deltac': c[i] - c[i - 1],
                    #                 'dc': [round(x, 3) for x in dc],
                    #             },
                    #             index=[i + k],
                    #         ),
                    #     ]
                    # )
                    r.loc[len(r)] = [
                        i + k,  # k from rough/fine searches
                        fail,
                        round(th[i][0], 3),
                        round(th[i][1], 3),
                        round(th[i][2], 3),
                        round(th[i][3], 3),
                        c[i],
                        c[i] - c[i - 1],
                        [round(x, 3) for x in dc],
                    ]

                    if cost < cmin:
                        cmin = cost
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # stopping condition
                    if patience_counter >= 50:
                        break

                    # stopping conditions
                    # if r[['deltac']][k:].dropna().rolling(100).std().iloc[-1,0] > 10:
                    #     break

                    # deltac = r[['deltac']].rolling(100).mean()[k:].dropna()
                    # if len(deltac[deltac.deltac > -0.1]) >= 50:
                    #     break

                    # if abs(c[i]-c[i-1])<0.3:
                    #     final_countdown -= 1

                    # if final_countdown == 0:
                    #     break

            #
            # Gradient descent small LR
            #

            # LR = (0.01,0.01,0.01)

            # print(f'{pd.Timestamp.now()} Gradient descent decrease LR to {LR} for {year_month} thresholds=({best_th0:.1f},{best_th1:.1f},{best_th2:.1f}), cost={best_cost:.1f} ')

            # j = i
            # for i in range(j,j+500):
            #     dc = grad_f(th[i-1])
            #     dth = [(dx+nz)*lr for dx,lr,nz in zip(dc,LR,rnd.rand(3))]
            #     new_th = [max(0,x+dx) for x,dx in zip(th[i-1],dth)]
            #     th.append(new_th)
            #     cost,fail,dispatch = f(th[i])
            #     c.append(cost)

            #     r.loc[len(r)] = [i+k,
            #                      fail,
            #                      round(th[i][0],3),
            #                      round(th[i][1],3),
            #                      round(th[i][2],3),
            #                      c[i],
            #                      c[i]-c[i-1],
            #                      [round(x,3) for x in dc]]

            best_cost = r[r.fail == False]['c'].min()
            imin = r[r.fail == False]['c'].idxmin()
            best_th0 = r.loc[imin].th0
            best_th1 = r.loc[imin].th1
            best_th2 = r.loc[imin].th2
            best_th3 = r.loc[imin].th3
            best_monthly.loc[len(best_monthly)] = [
                year_month,
                best_th0,
                best_th1,
                best_th2,
                best_th3,
                best_cost,
            ]

            t_gd = pd.Timestamp.now()
            dt = (t_gd - t_fgs).seconds
            print(
                f'{t_gd} (+{dt}s) Gradient descent done for {year_month} thresholds=({best_th0:.1f},{best_th1:.1f},{best_th2:.1f},{best_th3:.1f}), cost={best_cost:.1f} '
            )

        # best_monthly.index = best_monthly.month
        # best_monthly = best_monthly.drop(columns=['month'])

        # print('/// Best monthly')
        # print(f'Batt kwh {batt_kwh}')
        # print(f'Total cost {best_monthly.cost.sum():.1f}')
        # print(best_monthly.T,'\n\n')

        new_row = [angle, batt_kwh, best_monthly.cost.sum()]
        if cfg.grid_search_prices:
            new_row = [angle, batt_kwh, p_p, p_s, p_e2, p_e3, best_monthly.cost.sum()]
        for th0, th1, th2, th3 in zip(
            best_monthly.threshold0,
            best_monthly.threshold1,
            best_monthly.threshold2,
            best_monthly.threshold3,
        ):
            new_row.append((th0, th1, th2, th3))
        output.loc[len(output)] = new_row
        output.to_csv(outdir+'output.csv')

    # r.loc[r.fail==False,'c'] = r[r.fail==False].c
    # r.loc[r.fail==True,'c (fail)'] = r[r.fail==True].c
    # r[['c','c (fail)']].plot()
    print('Elapsed', pd.Timestamp.now() - tic)
    
    shutil.copyfile(cfg.filename, outdir+cfg.filename)
    
    if process_results and not cfg.grid_search_prices:
        pd.set_option('display.float_format', lambda x: f'{x:.1f}')
        results = process_output(df=output)
        results.index = results.index.rename('Battery [kWh]')
        results.to_csv(outdir+'results.csv')
        return results


# %%
if __name__ == '__main__':
    #
    # Config
    #
    cfg =   parse_inputs('caltech_ev.yaml')
    
    #
    # Data
    #
    tou =   TimeOfUseTariff(cfg.tou)
    load =  read_load_data(cfg)
    solar = read_and_scale_solar(cfg, load.index)
    df = calculate_net_load(load, solar)
    
    #
    # Single peak shave sim
    #
    """fail,dispatch,cost = peak_shaving_sim_4tou( df[['load','solar_w90','netload_w90']].loc['2018-6'],
                                                'netload_w90',
                                                soe_max=100,
                                                thresholds = (1e3,1e3,1e3,21),
                                                tou=tou.get(6),
                                                utility_chg_max=100,
                                                return_cost=True)"""

    
    #
    # Optimize
    #
    optimize_thresholds(cfg, df, tou , tou1_is_tou0=True, tou2_is_tou0=True, process_results=False)