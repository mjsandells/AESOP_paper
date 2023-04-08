import scipy.io as sio
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from smrt import make_snowpack, make_model, sensor_list, make_soil
from smrt.inputs.sensor_list import passive
from smrt.substrate.reflector import make_reflector
from smrt.core.globalconstants import DENSITY_OF_ICE, FREEZING_POINT

# Set up SMRT model configurations. dort normalized??
model = make_model('iba', 'dort',  rtsolver_options=dict(error_handling='nan'))

# Set up sensor configuration
# Need to get as near to nadir as possible to tie in with airborne observations
sensor_89 = sensor_list.passive(89e9, 0)
sensor_157 = sensor_list.passive(157e9, 0)
sensor_183 = sensor_list.passive(183e9, 0)
sensor_118 = sensor_list.passive(118e9, 0)
sensor_243 = sensor_list.passive(243e9, 0)

# Make a list of profiles i.e. separate measurement locations.
# Edit path to TVC2018 directory if required.
matfile = sio.loadmat('../Data/ground_obs/TVC2018_trenches.mat',struct_as_record=False, squeeze_me=True)
trenchdata = matfile['TVC']

profile_list = dir(trenchdata)[0:36]
profile_list = [s for s in profile_list if 'A1' not in s]

# Define soil substrate
soil = make_soil(substrate_model='flat',temperature=FREEZING_POINT-15, permittivity_model=complex(4, 0.5))


def m(value):
    '''Converts cm to m (for layer thickness)'''
    return value * 1e-2


def pex(ssa, density, mod_debye=0.75):
    radius = 3. / (ssa * DENSITY_OF_ICE)
    return mod_debye * 4 * (1. - density / DENSITY_OF_ICE) / (ssa * DENSITY_OF_ICE)


# May need adapting for profiles without sample B/2
# Define function to get max density from all layers
def max_density(profile, default_density):
    try:
        density = [max(np.fmax(profile.density.densityA, profile.density.densityB)[profile.density.density_layer_ID==l]) for l in np.unique(profile.density.density_layer_ID)]
    except:
        # No density B sample
        density = [np.nanmax(profile.density.densityA[profile.density.density_layer_ID==l])for l in np.unique(profile.density.density_layer_ID)]
    # Insert missing value if there are no data for a particular layer
    # List of layer IDs that should be present
    all_layers=[1, 2, 3]
    # Identify which are missing
    missing_layer = set(all_layers).difference(np.unique(profile.density.density_layer_ID))
    # Include missing layer properties from default
    for l in missing_layer:
        density.insert(l-1,default_density[l-1])
    # Set any NaN densities
    densitynans = np.argwhere(np.isnan(density)).flatten()
    for l in densitynans:
        density[l] = default_density[l]
    return np.asarray(density)


def min_density(profile, default_density):
    try:
        density = [min(np.fmin(profile.density.densityA, profile.density.densityB)[profile.density.density_layer_ID==l]) for l in np.unique(profile.density.density_layer_ID)]
    except:
        # No density B sample
        density = [np.nanmin(profile.density.densityA[profile.density.density_layer_ID==l]) for l in np.unique(profile.density.density_layer_ID)]
    # List of layer IDs that should be present
    all_layers=[1, 2, 3]
    # Identify which are missing
    missing_layer = set(all_layers).difference(np.unique(profile.density.density_layer_ID))
    # Include missing layer properties from default
    for l in missing_layer:
        density.insert(l-1,default_density[l-1])
    # Set any NaN densities
    densitynans = np.argwhere(np.isnan(density)).flatten()
    for l in densitynans:
        density[l] = default_density[l]
    return np.asarray(density)


def max_ssa(profile, default_ssa):
    try:
        ssa = [max(np.fmax(profile.SSA.SSA1, profile.SSA.SSA2)[profile.SSA.SSA_layer_ID==l]) for l in np.unique(profile.SSA.SSA_layer_ID)[~np.isnan(np.unique(profile.SSA.SSA_layer_ID))]]
        # np.isnan included to deal with nan in SSA_layer_ID in A05C
    except:
        # No SSA2 sample
        ssa = [np.nanmax(profile.SSA.SSA1[profile.SSA.SSA_layer_ID==l]) for l in np.unique(profile.SSA.SSA_layer_ID)]
    # List of layer IDs that should be present
    all_layers=[1, 2, 3]
    # Identify which are missing
    missing_layer = set(all_layers).difference(np.unique(profile.SSA.SSA_layer_ID))
    # Include missing layer properties from default
    for l in missing_layer:
        ssa.insert(l-1,default_ssa[l-1])
    # Set any NaN SSA
    ssanans = np.argwhere(np.isnan(ssa)).flatten()
    for l in ssanans:
        ssa[l] = default_ssa[l]
    return np.asarray(ssa)


def min_ssa(profile, default_ssa):
    try:
        ssa = [min(np.fmin(profile.SSA.SSA1, profile.SSA.SSA2)[profile.SSA.SSA_layer_ID==l]) for l in np.unique(profile.SSA.SSA_layer_ID)[~np.isnan(np.unique(profile.SSA.SSA_layer_ID))]]
        # np.isnan included to deal with nan in SSA_layer_ID in A05C
    except:
        # No SSA2 sample
        ssa = [np.nanmin(profile.SSA.SSA1[profile.SSA.SSA_layer_ID==l]) for l in np.unique(profile.SSA.SSA_layer_ID)]
    # List of layer IDs that should be present
    all_layers=[1, 2, 3]
    # Identify which are missing
    missing_layer = set(all_layers).difference(np.unique(profile.SSA.SSA_layer_ID))
    # Include missing layer properties from default
    for l in missing_layer:
        ssa.insert(l-1,default_ssa[l-1])
    # Set any NaN SSA
    ssanans = np.argwhere(np.isnan(ssa)).flatten()
    for l in ssanans:
        ssa[l] = default_ssa[l] 
    return np.asarray(ssa)


def flight_interpolated_temperatures(thicknesses, T_surface, T_base=263):
    # Function to interpolate temperatures - 263K assumed at base if not specified
    # This is mean of C087 and C090 basal temperatures - see check_flight_temperature_correction.ipynb
    # Calculate layer midpoint distance from surface
    layer_bases = np.cumsum(thicknesses)
    midpoints = [layer_bases[l]-thicknesses[l]/2 for l in range(len(thicknesses))]
    # Interpolate temperature from surface to base
    return np.interp(midpoints, [0, sum(thicknesses)], [T_surface, T_base])


def return_snowpack(profile, with_ice_lens=False, dh_debye=0.75, airgap=False, 
                    density_extreme=None, ssa_extreme=None, surface_crust=False,
                    depth_uncertainty=None, atmosphere=None, 
                    surface_snow_density=None, flight_temperature_correction=None):
    
        # Define soil substrate
    soil = make_soil(substrate_model='flat',temperature=FREEZING_POINT-15, permittivity_model=complex(4, 0.5))

    # Function to make snowpack:
    # deals with dual measurements and NaNs
    # inserts ice lens if needed

    ice_lens_density = 909 # Watts et al., TC 2016.
    ice_lens_ssa = 100 # REDEFINE WITH SENSIBLE VALUE
    default_density = np.asarray([103.7, 315.5, 253.1]) # From Rutter et al. 2019
    default_ssa = np.asarray([44.7, 23.8, 11.5])
    default_temp = 265. # Have asked Nick to fill in missing value so I don't have to use this

    # Get snow parameters for each layer
    layer_thickness = profile.layer_thickness.layer_thickness * 1e-2 # Convert cm to m
    densityA = profile.mean_layer_densityA.mean_layer_densityA
    # Don't always have densityB
    try:
        densityB = profile.mean_layer_densityB.mean_layer_densities
    except AttributeError:
        densityB = [np.nan] * 3
    density = np.nanmean(np.array([densityA, densityB]), axis=0) # Stacks density arrays, then calcs mean of 2 obs
    ssaA = profile.mean_layer_SSA1.mean_layer_SSA1
    try:
        ssaB = profile.mean_layer_SSA2.mean_layer_SSA2
    except AttributeError:
        ssaB = [np.nan] * 3
    ssa = np.nanmean(np.array([ssaA, ssaB]), axis=0) # Stacks ssa arrays, then calcs mean of 2 obs
    temperature = profile.mean_layer_temp.mean_layer_temp + FREEZING_POINT # Convert deg C to K
    
    # Fill in missing values in density / ssa / temperature if needed
    temperature[np.isnan(temperature)] = default_temp
    density[np.isnan(density)] = default_density[np.isnan(density)]
    ssa[np.isnan(ssa)] = default_ssa[np.isnan(ssa)]

    # Use max / min density / ssa if required
    if density_extreme is None:
        pass
    elif density_extreme == 'low':
        density = min_density(profile, default_density)
    elif density_extreme == 'high':
        density = max_density(profile, default_density)
    else:
        print ('density_extreme should be set to low, high or None')
        
    if ssa_extreme is None:
        pass
    elif ssa_extreme == 'low':
        ssa = min_ssa(profile, default_ssa)
    elif ssa_extreme == 'high':
        ssa = max_ssa(profile, default_ssa)
    else:
        print ('ssa_extreme should be set to low, high or None')   

    # Get rid of non-existent layers
    # Count number of layers from layer_thickness: nan means layer did not exist
    nlayers = np.count_nonzero(~np.isnan(layer_thickness))
    # Treat layer_thickness as truth i.e. if NaN then layer did not exist
    # Strip out NaN layers
    strip_layers = ~np.isnan(layer_thickness)
    layer_thickness = layer_thickness[strip_layers]
    density = density[strip_layers]
    ssa = ssa[strip_layers]
    temperature = temperature[strip_layers]
    
    # Modification of Debye relationship (Matzler 2002)
    mod_debye = np.ones(len(ssa)) * 0.75 # Generally used
    mod_debye[-1] = dh_debye # Adjust depth hoar modified Debye relationship
    
    # A04N has ice lens in middle of WS
    # Insert ice lens if needed and if present. Special cases A04C1 and A05W dealt with below
    if (with_ice_lens == True) & (isinstance(profile.ice_lens.ice_lens_thick,(int,float))):
        if m(profile.ice_lens.ice_lens_height_top) == layer_thickness[-1]:
            # At top of DH layer i.e. at WS-DH boundary
            layer_thickness[-1] -= m(profile.ice_lens.ice_lens_thick)
            layer_thickness = np.insert(layer_thickness,-1, m(profile.ice_lens.ice_lens_thick))
            density = np.insert(density, -1, ice_lens_density)
            ssa = np.insert(ssa, -1, ice_lens_ssa)
            mod_debye = np.insert(mod_debye, -1, 0.75)
            temperature = np.insert(temperature, -1, temperature[-1]) # Assume same temperature as DH layer
        elif m(profile.ice_lens.ice_lens_height_top) == layer_thickness[-2:].sum():
            # At top of WS layer i.e. at FS-WS boundary
            layer_thickness[-2] -= m(profile.ice_lens.ice_lens_thick)
            layer_thickness = np.insert(layer_thickness,-2, m(profile.ice_lens.ice_lens_thick))
            density = np.insert(density, -2, ice_lens_density)
            ssa = np.insert(ssa, -2, ice_lens_ssa)
            mod_debye = np.insert(mod_debye, -2, 0.75)
            temperature = np.insert(temperature, -2, temperature[-2]) # Assume same temperature as WS layer
        elif m(profile.ice_lens.ice_lens_height_top) < layer_thickness[-1]:
            # Ice lens is in depth hoar layer: need to split layer and insert lens
            layer_thickness[-1] -= m(profile.ice_lens.ice_lens_height_top) # Bottom layer is now thickness of DH above ice lens
            layer_thickness = np.append(layer_thickness, m(profile.ice_lens.ice_lens_thick)) # Add ice below
            layer_thickness = np.append(layer_thickness, m(profile.ice_lens.ice_lens_height_top - profile.ice_lens.ice_lens_thick))
            density = np.append(density, [ice_lens_density, density[-1]])
            ssa = np.append(ssa, [ice_lens_ssa, ssa[-1]])
            mod_debye = np.append(mod_debye, [0.75, dh_debye])
            temperature = np.append(temperature, [temperature[-1]]*2)
        elif m(profile.ice_lens.ice_lens_height_top) < layer_thickness[-2:].sum():
            # Ice lens is in wind slab: need to split layer and insert lens
            ice_lens_height_in_layer = m(profile.ice_lens.ice_lens_height_top) - layer_thickness[-1]
            layer_thickness[-2] -= ice_lens_height_in_layer # Wind slab thickness above ice lens
            layer_thickness = np.insert(layer_thickness,-1, m(profile.ice_lens.ice_lens_thick)) # Insert ice lens below
            layer_thickness = np.insert(layer_thickness, -1, ice_lens_height_in_layer - m(profile.ice_lens.ice_lens_thick))
            density = np.insert(density, -1, [ice_lens_density, density[-2]])
            ssa = np.insert(ssa, -1, [ice_lens_ssa, ssa[-2]])
            mod_debye = np.insert(mod_debye, -1, [0.75, 0.75])
            temperature = np.insert(temperature,-1, [temperature[-2]]*2)
        else:
            # Ice lens is in surface snow
            ice_lens_height_in_layer = m(profile.ice_lens.ice_lens_height_top) - layer_thickness[-2:].sum()
            layer_thickness[0] -= ice_lens_height_in_layer # FS thickness above ice lens
            layer_thickness = np.insert(layer_thickness,1, m(profile.ice_lens.ice_lens_thick)) # Insert ice lens below
            layer_thickness = np.insert(layer_thickness,2, ice_lens_height_in_layer - m(profile.ice_lens.ice_lens_thick))
            density = np.insert(density, 1, [ice_lens_density, density[0]])
            ssa = np.insert(ssa, 1, [ice_lens_ssa, ssa[0]])
            mod_debye = np.insert(mod_debye, 1, [0.75,0.75])
            temperature = np.insert(temperature,1, [temperature[0]]*2)
            
    # Need to do special modifications for pits A04C1 and A05W - ice lenses not in database
    # Identify from unique lat / lons (profile ID not passed in to this function)
    # A04C1
    if (with_ice_lens == True) & (profile.latitude.latitude == 68.7471) & (profile.longitude.longitude == 133.504441):
        # Need to insert 1.5mm ice lens at bottom of WS layer
        # A04C1 only has WS / DH layers
        # Reduce thickness of surface (WS) layer
        layer_thickness[0] -= 1.5e-3 # Reduce by 1.5mm
        layer_thickness = np.insert(layer_thickness,-1, 1.5e-3)
        density = np.insert(density, -1, ice_lens_density)
        ssa = np.insert(ssa, -1, ice_lens_ssa)
        mod_debye = np.insert(mod_debye, -1, 0.75)
        temperature = np.insert(temperature, -1, temperature[-1]) # Assume same temperature as DH layer       
        
    # A05W
    if (with_ice_lens == True) & (profile.latitude.latitude == 68.727063) & (profile.longitude.longitude == 133.534434):
        # Ice lens is in wind slab: need to split layer and insert lens
        ice_lens_height_in_layer = 0.39 - layer_thickness[-1]
        layer_thickness[-2] -= ice_lens_height_in_layer # Wind slab thickness above ice lens
        layer_thickness = np.insert(layer_thickness,-1, 1e-3) # Insert 1mm ice lens below
        layer_thickness = np.insert(layer_thickness, -1, ice_lens_height_in_layer - 1e-3)
        density = np.insert(density, -1, [ice_lens_density, density[-2]])
        ssa = np.insert(ssa, -1, [ice_lens_ssa, ssa[-2]])
        mod_debye = np.insert(mod_debye, -1, [0.75, 0.75])
        temperature = np.insert(temperature,-1, [temperature[-2]]*2)
    
    
    # PUT IN MODIFIED DEBYE IF ICE LENS IS FALSE
    
    # Strip out zero thickness layers
    # Specifically to deal with A03C1, where ice lens replaced very thin WS layer
    strip_layers = np.nonzero(layer_thickness)
    layer_thickness = layer_thickness[strip_layers]
    density = density[strip_layers]
    ssa = ssa[strip_layers]
    mod_debye = mod_debye[strip_layers]
    temperature = temperature[strip_layers]   

    corr_length = pex(ssa, density, mod_debye=mod_debye)

    if airgap == True:
        # Put air gap at bottom of pack
        # Assume: air gap is 5mm, SWE and layer thicknesses remain unchanged
        # This implies total height of snowpack is increased due to the added thickness
        # Also assume: very low density snow (10 kg m^-3) and low correlation length (1e-5m)
        # Also assume: temperature of air gap is same as bottom layer of snow
        layer_thickness = np.append(layer_thickness, 5e-3)
        density = np.append(density, 10)
        temperature = np.append(temperature, temperature[-1])
        corr_length = np.append(corr_length, 1e-5)
        
    if surface_crust is True:
        # Add a surface crust
        # NB CHECK ASSUMPTIONS
        layer_thickness = np.insert(layer_thickness, 0, 5e-3)
        density = np.insert(density, 0, ice_lens_density)
        temperature = np.insert(temperature, 0, temperature[0])
        corr_length = np.insert(corr_length, 0, 1e-5)

    if surface_snow_density is not None:
        layer_thickness = np.insert(layer_thickness, 0, 5e-2)
        density = np.insert(density, 0, surface_snow_density) # Should be provided in kg m^-3
        temperature = np.insert(temperature, 0, 260) # Low temperature
        corr_length = np.insert(corr_length, 0, 1e-4)
        
    if depth_uncertainty is None:
        pass
    elif depth_uncertainty == 'low':
        # Reduce depth hoar thickness by 2cm (minimum 1cm)
        layer_thickness[-1] = max((layer_thickness[-1] - 0.02), 0.01)
    elif depth_uncertainty == 'high':
        # Increase depth hoar thickness by 2cm
        layer_thickness[-1] = layer_thickness[-1] + 0.02
    else:
        print ('depth_uncertainty must be None, high or low')
        
    # Correct temperatures from pit measurements to time of flight:
    # Interpolate between air temperature and basal temperature (from pit)
    if flight_temperature_correction is None:
        pass
    elif flight_temperature_correction == 'C087':
        # Assume snow surface is -12.6 deg C
        temperature = flight_interpolated_temperatures(layer_thickness, -12.6 + FREEZING_POINT)
    elif flight_temperature_correction == 'C090':
        # Assume snow surface is -15.4 deg C
        temperature = flight_interpolated_temperatures(layer_thickness, -15.4 + FREEZING_POINT)
    elif flight_temperature_correction == 'C092':
        # Assume snow surface is -14.1 deg C
        temperature = flight_interpolated_temperatures(layer_thickness, -14.1 + FREEZING_POINT)
    else:
        print ('flight_temperature_correction not recognised. Should be "C087", "C090", "C092" or "None". No correction applied')
    
    return make_snowpack(layer_thickness, "exponential", density=density, corr_length=corr_length,
                                       temperature=temperature, substrate=soil, atmosphere=atmosphere)

