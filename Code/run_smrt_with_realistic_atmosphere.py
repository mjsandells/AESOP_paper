import glob
import pyarts
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import math
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import aesop
from smrt import sensor_list
from smrt_realistic_atmosphere import RealisticAtmosphere
from smrt_aesop_setup import model, trenchdata, return_snowpack, sensor_89, sensor_157, sensor_183, sensor_118, sensor_243


def get_atmosphere(flight, aoi, single_frequency=False):
    '''
    Returns ARTS atmosphere object for requested flight and AOI containing all frequencies.

    Args:
        flight (str): flight number to determine which profiles to use.
        aoi (str): profile and average ground height returned for requested aoi.
    '''
    altitudes = aesop.mean_alt(flight)

    channels = {'89': [88992000000.0, 1075000000.0, 650000000.0],
                '118+5.0': [118750000000.0, 5000000000.0, 2000000000.0],
                '157': [157075000000.0, 2600000000.0, 2600000000.0],
                '183+7': [183248000000.0, 7000000000.0, 2000000000.0],
                '243': [243200000000.0, 2500000000.0, 3000000000.0]}
    
    profile = ('../Data/retrieved_profiles/'+flight+'/'+flight+'_'+aoi+'.xml')
    sensor_height = altitudes[aoi]
    ground_height = pyarts.xml.load(profile)[0,2]
        
    if single_frequency == True:
        atmosphere = RealisticAtmosphere(profile,sensor_height,ground_height)
    else:
        atmosphere = RealisticAtmosphere(profile,sensor_height,ground_height,channel=channels)

    return atmosphere


def get_snowpacks(flight, surface_snow_density=None, with_ice_lens=True):
    '''
    Returns dictionary of snowpacks with atmosphere for the base, high and low snowpacks.
       
    Args:
        flight (str): flight number to determine which atmospheric profiles to use.
    '''
    pits = aesop.get_pits()
    pits = pits[~pits.index.str.contains('A1')].rename(index={'MetStation': 'MetS'})
    
    base_snow = {}
    high_snow = {}
    low_snow = {}
    
    for p in pits.index:
        profile = eval('trenchdata' + '.' + p)
        if p == 'MetS':
            aoi = 'A04'
        else:
            aoi = p[:3]
                        
        atmosphere = get_atmosphere(flight, aoi)
        
        base_snow.update({p: return_snowpack(profile, with_ice_lens=with_ice_lens, dh_debye=1.2,
                                             surface_snow_density=surface_snow_density,
                                             atmosphere=atmosphere,
                                             flight_temperature_correction=flight)})
        high_snow.update({p: return_snowpack(profile, with_ice_lens=with_ice_lens, density_extreme='high',
                                             ssa_extreme='high', dh_debye=1.2,
                                             surface_snow_density=surface_snow_density,
                                             atmosphere=atmosphere,
                                             flight_temperature_correction=flight)})
        low_snow.update({p: return_snowpack(profile, with_ice_lens=with_ice_lens, density_extreme='low',
                                            ssa_extreme='low', dh_debye=1.2,
                                            surface_snow_density=surface_snow_density,
                                            atmosphere=atmosphere,
                                            flight_temperature_correction=flight)})
        
        
    return base_snow, high_snow, low_snow


def run_smrt_with_atmosphere(flight, extremes=False, fresh_snow=None, with_ice_lens=True, skip_atmos=False):
    '''
    Runs SMRT for each pit snowpack including the ARTS atmosphere. Returns a Pandas DataFrame of
    SMRT result objects.

    Args:
        flight (str): flight number to determine which atmospheres to use.
        extremes (bool): if True returns results for high and low snowpacks as well as base,
                         default is False which only returns base snowpacks
        fresh_snow (float): If set, will add 5cm fresh snow layer to the surface of all profiles
                            with density equal to the value given. 
                            40 kg m^-3 recommended from pit A03C1 measured on 17th March
        with_ice_lens (bool): adjust snowpack to contain ice lenses where they exist (default: True)
    '''
    start = datetime.now()

    base_snow, high_snow, low_snow = get_snowpacks(flight, surface_snow_density=fresh_snow, with_ice_lens=with_ice_lens)
    
    channels = list(base_snow.values())[0].atmosphere.channel.values()
    frequencies = np.array(list(channels))[:,0]
    sensors = sensor_list.passive(frequencies, 0)
    
    # Neglect atmosphere if required (figure 10)
    if skip_atmos is True:
        for k in base_snow.keys():
            base_snow[k].atmosphere = None 
            high_snow[k].atmosphere = None 
            low_snow[k].atmosphere = None 

    results = {'base': model.run(sensors, base_snow)}
    
    if extremes == True:
        results.update({'high': model.run(sensors, high_snow),
                        'low': model.run(sensors, low_snow)})

    print(datetime.now()-start)
    return results


def tb_results(results, save=False, filename=None, single_frequency=None):
    '''
    Extracts TbV results and corrects to Rayleigh-Jeans TB to match observations.

    Args:
        results (DataArray): SMRT results
        save (bool): If True saves results to netCDF in specified file
        filename (str): path to netCDF file to save results
    '''
    
    # Single result, single frequency correction
    if isinstance(results, float) and single_frequency is not None:
        offset = (scipy.constants.h*single_frequency)/(2*scipy.constants.k) # correct for Rayleigh-Jeans TB
        return (results - offset)
    
    
    tb_res = {}

    for snowpacks, result in results.items():
        tb = []
        for res in result.TbV():
            freq = res.frequency.values
            offset = (scipy.constants.h*freq)/(2*scipy.constants.k) # correct for Rayleigh-Jeans TB
            tb.append(res-offset)

        tb_res.update({snowpacks: xr.concat(tb, dim='frequency')})

    tb_results = xr.Dataset(tb_res)
    
    # Add instrument channels names to match those in observations
    channels = ['M16-89','118+5.0','M17-157','M20-183+7','243']
    tb_results = tb_results.assign_coords(channel=('frequency', channels))
    tb_results = tb_results.swap_dims({'frequency':'channel'})

    if save == True:
        tb_results.to_netcdf(filename)

    return xr.Dataset(tb_results)


def ground_based_data_corrected(data, channel, pits):
    '''Returns ground based data corrected for the atmospheric upwelling and transmission.'''
    tb_ground = []

    if channel == '89H0':
        c1 = 'TB89H_0'
        c2 = 'TBstd89H_0'
        theta = 0
    elif channel == '89V0':
        c1 = 'TB89V_0'
        c2 = 'TBstd89V_0'
        theta = 0
    elif channel == '89H55':
        c1 = 'TB89H_55'
        c2 = 'TBstd89H_55'
        theta = 55
    elif channel == '89V55':
        c1 = 'TB89V_55'
        c2 = 'TBstd89V_55'
        theta = 55
    else:
        print ('Channel not recognized: expecting 89H0, 89V0, 89H55 or 89V55')
        return

    for p in pits.index:
        if p[-1] != '1':
            p = p + '1'
            
        if 'Met' in p:
            aoi = 'A04'
        else:
            aoi = p[:3]
            
        atmosphere = get_atmosphere('C087', aoi, True)

        if p in data.Site.unique():
            tbup = atmosphere.tbup(89e9, math.cos(np.radians(theta)), 2)[0]
            trans = atmosphere.trans(89e9, math.cos(np.radians(theta)), 2)[0]
            
            data_corrected = data[data.Site==p][c1].apply(lambda x: (x*trans) + tbup)

            tbmean = data_corrected.mean()

            if np.isfinite(tbmean):
                errmin = tbmean - data_corrected.min() + data[data.Site==p][c2][data_corrected.idxmin()]
                errmax = data_corrected.max() + data[data.Site==p][c2][data_corrected.idxmax()] - tbmean
            else:
                errmin = np.nan
                errmax = np.nan
            tb_ground.append([tbmean, errmin, errmax])
        else:
            tb_ground.append([np.nan]*3)

    return np.asarray(tb_ground)
