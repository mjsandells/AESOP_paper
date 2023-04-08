import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import iris
import scipy
from scipy import constants
import math
import rasterio
from rasterstats import gen_point_query


def distance(lats, lons, pitlat, pitlon):
    '''
    Set up functions to determine closest points with Haversine formula
    https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude
    lats, lons can be array of points, pitlat, pitlon an individual point
    '''
    p = 0.017453292519943295
    a = 0.5 - np.cos((pitlat-lats)*p)/2 + np.cos(lats*p)*np.cos(pitlat*p) * (1-np.cos((pitlon-lons)*p)) / 2
    # Dist in km. 0.5% error due to non-spherical Earth https://en.wikipedia.org/wiki/Haversine_formula
    return 12742 * np.arcsin(np.sqrt(a))


def extract_flight_data_for_pit(subset, pit, chan):
    '''
    Function to extract all flight data close enough to a particular pit
    Channel 0 is 89 GHz
    '''
    TB = []
    for fl in pit:
        # Pull out distribution of 89GHz (channel 0) TB for flight data within max_dist of each pit
        TB.append([subset.sel(time=fl, channel=subset.channel[chan]).brightness_temperature.values,
              subset.sel(time=fl).bs_latitude.values, subset.sel(time=fl).bs_longitude.values])
    return np.vstack(TB)


def get_pits():
    '''
    Get snowpit info, put date, lat and long into list
    Nick and Richard's pits
    '''
    pitfiles = glob.glob("../Data/ground_obs/Pit_notes/*.xlsx")
    index = [] # Pit names, for pandas dataframe
    pitmeta = []
    for p in pitfiles:
        pitnotes = pd.read_excel(p, sheet_name='PIT', header=None)
        time = datetime.strptime(pitnotes[14][5], '%H:%M')
        # Lat in column 7, row 1; long in column7, row 3; date in col 12, row 5; time in col 14, row 5
        pittime = pitnotes[12][5] + timedelta(hours=time.hour, minutes=time.minute)
        # Date, lat, long (convert to -ve)
        pitmeta.append([pittime, pitnotes[7][1], -pitnotes[7][3]])
        # Store pit names
#         index.append(p[38:42])
        index.append(p[29:-5])
    # Convert to pandas
    pits = pd.DataFrame(pitmeta, columns=['date', 'lat', 'long'], index=index)

    # Get Sherbrook pit locations
    pitfiles = glob.glob("../Data/ground_obs/Sherbrooke_Pit_SSA/*.xlsx")
    index = [] # Pit names, for pandas dataframe
    pitmeta = []
    for p in pitfiles:
        pitnotes = pd.read_excel(p, sheet_name='Site', header=None)
        pittime = (datetime.strptime(p[-18:-5], '%Y%m%d_%H%M'))
        # Lat: col 12, row 1; Long: col 13, row 1
        pitmeta.append([pittime, pitnotes[12][1], pitnotes[13][1]])
#         index.append(pitnotes[0][1]) # ID code for pit
        index.append(p[42:-19]) # ID code for pit

    sherbrookepits = pd.DataFrame(pitmeta, columns=['date', 'lat', 'long'], index=index)

    # Join two sets of pits
    pits = pits.append(sherbrookepits).sort_index()
    
    for pit in pits.index:
        for aoi in ['A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']:
            if aoi in pit:
                pits.loc[pit, 'aoi'] = aoi
        if pit == 'MetStation':
            pits.loc[pit, 'aoi'] = 'A04'
        
    return pits


def find_flights_near_pits(subset, channel, pits, max_dist_in_km):
    '''
    Find all flight observations within close spatial proximity to pits
    Store observations time and use as search term
    Channel 0: 89
    Channel 1: 157
    '''
    nearpits = []
    for p in range(len(pits)):
        pit_distance = distance(subset.bs_latitude.astype('float64'), subset.bs_longitude.astype('float64'), 
                                pits.lat.values[p], pits.long.values[p])
        nearpits.append(pit_distance[pit_distance<max_dist_in_km].time.values)    
    # Get flight TB for each pit
    TB_all = []
    for pit in nearpits:
        try:
            TB_all.append(extract_flight_data_for_pit(subset, pit, channel)[:,0])
        except:
            TB_all.append(np.nan)
    return TB_all


def test_algorithm(subset, pits, max_dist_in_km):
    '''
    Test algorithm
    Store observations time and use as search term
    Channel 0: 89
    Channel 1: 157
    '''
    nearpits = []
    for p in range(len(pits))[:1]:
        pit_distance = distance(subset.bs_latitude.astype('float64'), subset.bs_longitude.astype('float64'), 
                                pits.lat.values[p], pits.long.values[p])
        nearpits.append(pit_distance[pit_distance<max_dist_in_km].time.values)    
    coords = []
    for fl in nearpits: # flight time
        # Pull out distribution of 89GHz (channel 0) TB for flight data within max_dist of each pit
        coords.append([subset.sel(time=fl).bs_latitude.values, subset.sel(time=fl).bs_longitude.values])
    return coords


def pit_flighttimes(subset, pits, max_dist_in_km):
    nearpits = []
    for p in range(len(pits)):
        pit_distance = distance(subset.bs_latitude.astype('float64'), subset.bs_longitude.astype('float64'), 
                                pits.lat.values[p], pits.long.values[p])
        nearpits.append(pit_distance[pit_distance<max_dist_in_km].time.values)    
    return nearpits


def ground_based_data(data, channel, pits):
    '''Function to get data for all pits, single channel'''
    tb_ground = []
    # Identify column names
    if channel == '89H0':
        c1 = 'TB89H_0'
        c2 = 'TBstd89H_0'
    elif channel == '89V0':
        c1 = 'TB89V_0'
        c2 = 'TBstd89V_0'
    elif channel == '89H55':
        c1 = 'TB89H_55'
        c2 = 'TBstd89H_55'
    elif channel == '89V55':
        c1 = 'TB89V_55'
        c2 = 'TBstd89V_55'
    else:
        print ('Channel not recognized: expecting 89H0, 89V0, 89H55 or 89V55')
        return
        
    for p in pits.index:
        # For UK pits, search for CA observations: need to add '1' to site name
        if p[-1] != '1':
            p = p + '1'
        if p in data.Site.unique():
            tbmean = data[data.Site==p][c1].mean()
            if np.isfinite(tbmean):
                errmin = tbmean - data[data.Site==p][c1].min() + data[data.Site==p][c2][data[data.Site==p][c1].idxmin()]
                errmax = data[data.Site==p][c1].max() + data[data.Site==p][c2][data[data.Site==p][c1].idxmax()] - tbmean
            else:
                errmin = np.nan
                errmax = np.nan
            tb_ground.append([tbmean, errmin, errmax])
        else:
            tb_ground.append([np.nan]*3)
            
    return np.asarray(tb_ground)


def get_marss_data(max_dist_in_km, theta, channel, pits):
    '''Function to get MARSS data within specified distance of pits'''
    subset = subset_marss(theta)
    return find_flights_near_pits(subset, channel, pits, max_dist_in_km)


def subset_marss(theta):
    # Get list of datafiles
    datafiles = glob.glob("../Data/raw_airborne/*marss*.nc")
    # Reads in all data from flights, picks out those with view angle
    for f, file in enumerate(datafiles):
        # get data
        data = xr.open_dataset(file)
        if f == 0:
            subset = data.where((data.sensor_view_angle > theta-5) & 
                                    (data.sensor_view_angle<theta+5), drop=True)
        else:
            subset = xr.concat([subset, data.where((data.sensor_view_angle > theta-5) & 
                                    (data.sensor_view_angle<theta+5), drop=True)], dim='time')
    # Need to drop all data with NaN for bs_latitude, bs_longitude
    return subset.dropna(dim='time')


def get_ismar_data(max_dist_in_km, theta, channel, pits):
    '''Function to get ISMAR data within specified distance of pits'''
    subset = subset_ismar(theta)
    return find_flights_near_pits(subset, channel, pits, max_dist_in_km)


def subset_ismar(theta):
    # Get list of datafiles
    datafiles = glob.glob("../Data/raw_airborne/*ismar*.nc")
    # Reads in all data from flights, picks out those with view angle
    for f, file in enumerate(datafiles):
        # get data
        data = xr.open_dataset(file)
        if f == 0:
            subset = data.where((data.sensor_view_angle > theta-5) & 
                                    (data.sensor_view_angle<theta+5), drop=True)
        else:
            subset = xr.concat([subset, data.where((data.sensor_view_angle > theta-5) & 
                                    (data.sensor_view_angle<theta+5), drop=True)], dim='time')
    # Need to drop all data with NaN for bs_latitude, bs_longitude
    return subset.dropna(dim='time')


def extract_emissivity(emdata, overpasstimes, channel):
    '''
    emdata are concatenated emissivity transects
    fltimes is flighttimes within 250m of pits. e.g. overpass_times[0] where 0 is pit number
    channel is 0 to 4
    '''
    emtimes = []
    for et in emdata.coord('time').points:
        emtimes.append(emdata.coord('time').units.num2date(np.floor(et)) + timedelta(seconds = np.floor(1e3 * (et %1)) / 1e3))
    emiss = []
    for op in overpasstimes:
        # Only extract emissivity if flight times are in emissivity file
        # this is needed as emissivities are only calculated over transects, whereas
        # flight times includes observations taken during turning circles
        if np.datetime64(op, 'ms').astype(datetime) in emtimes:
            # Find where it is
            index = list(emtimes).index(np.datetime64(op, 'ms').astype(datetime))
            # look up emissivity for that channel and append
            emiss.append(emdata[index][channel].data)

    return np.asarray(emiss).flatten()      


def cat_transect(filestring):
    '''Concatenate transect cubes'''
    filesearch = '../retrieved_emissivity/*' + str(filestring) + '*.nc'
    # Get list of datafiles
    datafiles = glob.glob(filesearch)
    #for f, file in enumerate(datafiles):
    #    print (f, file)
    cubes = iris.load(datafiles, 'surface_microwave_emissivity')
    # Fix attributes (file date stamp) to be same so can concatenate !!! BE CAREFUL
    if cubes[0].long_name == 'retrieved surface emissivity': # Iris does not seem to always retrieve cubes in same order each time
        start = 0
    else:
        start = 1
    cubes = cubes[start::2] # Ignores half the cubes, which are emissivity retrieved at retrieval frequency
    for c in range(len(cubes)):
        cubes[c].attributes = cubes[0].attributes
    return cubes.concatenate()[0]


def bt_offset(channel):
    '''
    Returns offset (hv/2k) to convert brightness temperatures to 
    Rayleigh–Jeans brightness temperature to match observations.
    
    Args
        channel (str): offsets are frequency dependent
    
    Returns
        offset (float): temperature offset for Rayleigh_Jeans
    '''
    offset = 0
    if channel in ['M16-89']:
        offset = (scipy.constants.h*89e9)/(2*scipy.constants.k)
    elif channel in ['M17-157']:
        offset = (scipy.constants.h*157e9)/(2*scipy.constants.k)
    elif channel in ['M18-183+1', 'M19-183+3', 'M20-183+7']:
        offset = (scipy.constants.h*183e9)/(2*scipy.constants.k)
    elif channel in ['118+1.1', '118+1.5', '118+2.1', '118+3.0', '118+5.0']:
        offset = (scipy.constants.h*118e9)/(2*scipy.constants.k)
    elif channel in ['243']:
        offset = (scipy.constants.h*243e9)/(2*scipy.constants.k)
    return offset


def slf_data(file, theta):
    '''
    Returns data for straight and level flight at requested view angle.Only getting data below 600ft,
    the max altitude for low level TVC flights.
    
    Args
        file (path): MARSS or ISMAR data file
        theta (int/float): view angle for selecting data
    '''
    data = xr.open_dataset(file)
    return data.where((data.platform_roll_angle<5)&(data.platform_roll_angle>-5)&
                      (data.sensor_view_angle > theta-5)&(data.sensor_view_angle<theta+5)&
                      (data.altitude<=600), drop=True)


def mean_alt(flight):
    '''Returns mean altitude for each AOI.'''
    if flight == 'C087':
        aois,_ = get_aoi_times()
        data = slf_data('../Data/raw_airborne/metoffice-marss_faam_20180316_r001_c087.nc', 0)
        
        alt = pd.Series([np.mean(data.altitude.sel(time=time).values) for time in aois.time],
                        index=aois.index)
        
    elif flight == 'C090':
        _,aois = get_aoi_times()
        data = slf_data('../Data/raw_airborne/metoffice-marss_faam_20180320_r001_c090.nc', 0)
        
        mean_alts = []
        for run in ['run1', 'run2']:
            mean_alts.append([np.mean(data.altitude.sel(time=time).values) for time in aois[run].time])
        alt = pd.Series(np.mean(mean_alts, axis=0), index=aois.index)
        
    return alt


def mean_ground_height(flight):
    '''Returns mean ground height for each AOI.'''
    if flight == 'C087':
        aois,_ = get_aoi_times()
        data = slf_data('../Data/raw_airborne/metoffice-marss_faam_20180316_r001_c087.nc', 0)
        
        mean_heights = [np.mean((data.altitude.sel(time=time).values)-(data.height.sel(time=time).values))
                  for time in aois.time]
        ground = pd.Series(mean_heights, index=aois.index)

        
    elif flight == 'C090':
        _,aois = get_aoi_times()
        data = slf_data('../Data/raw_airborne/metoffice-marss_faam_20180320_r001_c090.nc', 0)
        
        mean_heights = []
        for run in ['run1', 'run2']:
            mean_heights.append([float(np.mean(data.altitude.sel(time=time).values-data.height.sel(time=time).values))
                      for time in aois[run].time])

        ground = pd.Series(np.mean(mean_heights, axis=0), index=aois.index)
        
    return ground


def extend_aoi_time(aois):
    '''Extend AOI times by 30 seconds to get downwelling data either side of the AOI.'''
    extended = {}
    for aoi, time in aois.iterrows():
        extended.update({aoi: slice(time.time.start-timedelta(seconds=30), time.time.stop+timedelta(seconds=30))})
    return pd.Series(extended)


def obs_per_view_angle(file, aois, instrument):
    '''
    Returns mean view angle and mean downwelling brightness temperature for multiple channels,
    at multiple scan angles, for each AOI.

    Args
        file: MARSS or ISMAR data file for flight
        aois (DataFrame): AOI times for flight (if C090 specific which run e.g. C090_aois.run1)
        instrument (str): State whether data if from MARSS or ISMAR to determine channels
                          and angles.
     
     Returns
        result (DataFrame): DataFrame with lists of view angles and downwelling brightness temperatures
                            for each AOI.
    '''
    if instrument == 'marss':
        channels = [0,1,2,3,4]
        channel_names = ['M16-89','M17-157','M18-183+1','M19-183+3','M20-183+7']
        angles = [0, 10, 20, 30, 40, 350, 340, 330, 320]
    elif instrument == 'ismar':
        channels = [0,1,2,3,4]
        channel_names = ['118+1.1','118+1.5','118+2.1','118+3.0','118+5.0']
        angles = [0, 10, 350, 340, 330, 320]

    
    result = pd.DataFrame(index = aois.index, columns=['angles']+channel_names)
    data = xr.open_dataset(file)

    extended_aois = extend_aoi_time(aois)
    for aoi, time in extended_aois.iteritems():
        view_angles = [np.nanmean([180-x for x in data.sel(time=time).sensor_view_angle.where(data.angle==angle)]) for angle in angles]

        sort = np.argsort(view_angles)
        result['angles'][aoi] = list(np.array(view_angles)[sort])

        for channel, channel_name in zip(channels, channel_names):
            tb = [np.nanmean(data.sel(time=time).brightness_temperature.where(data.angle==angle)[:,channel]) for angle in angles]

            result[channel_name][aoi] = list(np.array(tb)[sort])
            
    return result


def get_aoi_polygons():
    latlon = pd.read_csv('../Data/AOIS/XY_Lat_Lons_Polygons.csv',header=None).values.reshape(-1,10)
    
    # Generate shapely polygons for each AOI and add to a GeoDataFrame excluding lake ice AOIS:
    polygons = []
    for row in latlon[1:]:
        polygons.append(Polygon(row.reshape(-1,2)))
    aoi_polygons = gpd.GeoDataFrame(geometry=polygons[0:8], index=['A02', 'A07', 'A03', 'A05', 'A04', 'A06', 'A08', 'A09'])
    
    # Set crs of AOI polygon GeoDataFrame - it is necessary to set a crs before you can transform to another crs:
    aoi_polygons.crs = {'init':'epsg:4326'}
    
    # Update A05 with extended polygon
    A05_points = [(559035.262177481, 7624809.443404101),
                  (560476.650386034, 7625749.995791388),
                  (560784.8133532323, 7625280.007221624),
                  (559341.068395522, 7624339.534490865),
                  (559035.262177481, 7624809.443404101)]
    
    new_A05 = gpd.GeoSeries(Polygon(A05_points))
    new_A05.crs = {'init':'epsg:26908'}
    
    aoi_polygons.loc['A05'] = new_A05.to_crs(epsg=4326)[0]
    
    return aoi_polygons


def get_pit_topography():
    topography_rast = rasterio.open('../Data/topography/TPIsimple.tif')

    # Get pits from shapefile and reproject to crs of raster.
    pits = gpd.read_file('../Data/topography/Pits_2013_2018.shp').to_crs(crs=topography_rast.crs.data)
    # Drop 2013 pits and lake ice pits.
    pits = pits[~pits.Location_I.str.contains('|'.join(['PIT', 'A1']))].reset_index(drop=True)
    pits = pits.drop(['Field4','Field5'], axis=1)

    # Use rasterstats gen_point_query to get topography type for each pit:
    # - gen_point_query returns a generator object so need to list each element
    # - Needs the path to the TIF rather than the TIF that has been read in

    topo = [t for t in gen_point_query(pits, '../Data/topography/TPIsimple.tif', interpolate='nearest')]

    pits['topo_id'] = topo
    pits['topo_type'] = pits['topo_id'].map({1.0:'plateau', 2.0:'valley', 3.0:'slopes'})

    pits = pits.set_index('Location_I').sort_index()

    return pits


def aoi_polygon_time(flight_data, E1_E2, E3_E4, E7_E8):
    aoi_polygons = get_aoi_polygons()
    
    aoi_time = gpd.GeoDataFrame(columns=['time'])
    for leg, aois in zip([E1_E2, E3_E4, E7_E8], [['A03', 'A05', 'A08', 'A09'], ['A07', 'A02'], ['A04', 'A06']]):
        data = flight_data.sel(time=leg)
        data_gdf = gpd.GeoDataFrame(data.time.values, geometry=gpd.points_from_xy(data.longitude, data.latitude), columns=['time'])
        for aoi, polygon in aoi_polygons.loc[aois].iterrows():
            times = [row.time for _,row in data_gdf.iterrows() if row.geometry.within(polygon.geometry)]
            aoi_time.loc[aoi] = slice(times[0], times[-1])
    return aoi_time


def get_aoi_times():
    '''
    Time slice for each leg in flights C087 and C090 from FAAM flight summary files
    ensure aoi times reflect individual overpasses of the legs.
    '''
    C087_marss = '../Data/raw_airborne/metoffice-marss_faam_20180316_r001_c087.nc'
    C087_ismar = '../Data/raw_airborne/metoffice-ismar_faam_20180316_r001_c087.nc'

    C090_marss = '../Data/raw_airborne/metoffice-marss_faam_20180320_r001_c090.nc'
    C090_ismar = '../Data/raw_airborne/metoffice-ismar_faam_20180320_r001_c090.nc'
    
    data_c087 = xr.open_dataset(C087_marss)
    C087_E1_E2 = slice(datetime(2018,3,16,22,23,1), datetime(2018,3,16,22,26,15))
    C087_E3_E4 = slice(datetime(2018,3,16,22,29,17), datetime(2018,3,16,22,31,1))
    C087_E7_E8 = slice(datetime(2018,3,16,22,42,48), datetime(2018,3,16,22,44,44))

    data_c090 = xr.open_dataset(C090_marss)
    C090_E1_E2_1 =  slice(datetime(2018,3,20,20,0,43), datetime(2018,3,20,20,4,25))
    C090_E3_E4_1 = slice(datetime(2018,3,20,20,8,43), datetime(2018,3,20,20,10,21))
    C090_E7_E8_1 = slice(datetime(2018,3,20,20,22,51), datetime(2018,3,20,20,25,12))
    C090_E1_E2_2 = slice(datetime(2018,3,20,20,29,29), datetime(2018,3,20,20,32,53))
    C090_E3_E4_2 = slice(datetime(2018,3,20,20,35,50), datetime(2018,3,20,20,37,28))
    C090_E7_E8_2 =  slice(datetime(2018,3,20,20,50,5), datetime(2018,3,20,20,52,6))

    aoi_times_c087 = aoi_polygon_time(data_c087, C087_E1_E2, C087_E3_E4, C087_E7_E8)
    
    aoi_times_c090_run1 = aoi_polygon_time(data_c090, C090_E1_E2_1, C090_E3_E4_1, C090_E7_E8_1)
    aoi_times_c090_run2 = aoi_polygon_time(data_c090, C090_E1_E2_2, C090_E3_E4_2, C090_E7_E8_2)
    aoi_times_c090 = pd.concat([aoi_times_c090_run1, aoi_times_c090_run2], axis=1, keys=['run1', 'run2'])
    
    return aoi_times_c087, aoi_times_c090


def obs_subset(flight):
    C087_marss = '../Data/raw_airborne/metoffice-marss_faam_20180316_r001_c087.nc'
    C087_ismar = '../Data/raw_airborne/metoffice-ismar_faam_20180316_r001_c087.nc'

    C090_marss = '../Data/raw_airborne/metoffice-marss_faam_20180320_r001_c090.nc'
    C090_ismar = '../Data/raw_airborne/metoffice-ismar_faam_20180320_r001_c090.nc'
    
    if flight == 'C087':
        marss = slf_data(C087_marss, 0)
        ismar = slf_data(C087_ismar, 0)

    elif flight == 'C090':
        C090 = slice(datetime(2018,3,20,20,0,43), datetime(2018,3,20,20,52,6)) # Only take the first two runs of C090
        marss = slf_data(C090_marss, 0).sel(time=C090)
        ismar = slf_data(C090_ismar, 0).sel(time=C090)

    obs_df = {}
    for channel in [marss.channel.values[i] for i in (0,1,4)]+list(ismar.channel.values[4:6]):
        data = marss if 'M' in str(channel) else ismar
        obs_df.update({str(channel)[2:-1]:
            gpd.GeoDataFrame(data.brightness_temperature.sel(channel=channel),
                             geometry=gpd.points_from_xy(data.longitude, data.latitude),
                             columns=['TB'], crs={'init':'epsg:4326'}).dropna().reset_index(drop=True)})

    return pd.concat(obs_df, axis=1).rename(columns={'243-H':'243'})


def get_obs_per_aoi(flight):
    obs_df = obs_subset(flight)
    aoi_polygons = get_aoi_polygons()
    topography_rast = rasterio.open('../Data/topography/TPIsimple.tif')

    obs_aois = {}
    for channel in obs_df.columns.levels[0]:
        obs = obs_df[channel].dropna()

        aoi_obs = pd.concat([obs.loc[obs.geometry.within(polygon)].reset_index(drop=True) for polygon in aoi_polygons.geometry],
                            keys=aoi_polygons.index).to_crs(crs=topography_rast.crs.data)

        for aoi, row in aoi_obs.groupby(level=0):
            topo = [t for t in gen_point_query(row.geometry, '../Data/topography/TPIsimple.tif', interpolate='nearest')]

            aoi_obs.loc[aoi, 'topo_id'] = topo
            aoi_obs['topo_type'] = aoi_obs['topo_id'].map({1.0:'plateau', 2.0:'valley', 3.0:'slopes'})

        obs_aois.update({channel: aoi_obs})
    return pd.concat(obs_aois, axis=1)
