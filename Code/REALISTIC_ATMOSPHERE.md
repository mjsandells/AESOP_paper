# Realistic Atmosphere
The realistic atmosphere module uses the ARTS radiative transfer model to calculate
the downwelling brightness temperature at the surface, and the absorption and emission
characteristics of the atmospheric layer between the surface and the sensor. It assumes 
clear-sky conditions and accounts for absorption by oxygen, nitrogen and water vapour. 
An input atmospheric profile is required to define the atmospheric state.

## Requirements
Use of the realistic atmosphere module requires the following:
 * Python version 3.6 or higher
 * The ARTS radiative transfer model <https://www.radiativetransfer.org>
 
ARTS must be built with the C API enabled, and pyarts must be properly configured.

## Using the realistic atmosphere
The realistic atmosphere needs to be configured with the atmospheric
profile, sensor height and ground height. The profile can either
be the path to an ARTS xml file containing a `Matrix` with the profile data, or a NumPy `array`. The columns are pressure (Pa), temperature (K), height (m) and water vapour volume mixing ratio, and the data must be
in order of decreasing pressure. For example:
```python
atm = RealisticAtmosphere("path/to/profile.xml", 300.0, 0.0)
```
will create an atmosphere using the given file with the sensor at 300m and the ground at 0m. Note that if the requested ground height is below the lowest height in the input profile then the lowest height in the profile will be used as the ground height.

Some microwave sensors have a dual-sideband configuration, for example to make measurements around atmospheric absorption lines. For such sensors it can be desirable to run SMRT using the line centre frequency, but the atmospheric contribution must be calculated across the actual sensor passbands. To handle these sensors the realistic atmosphere must be configured with the appropriate passband information. Note that the atmosphere will then ignore any frequency information passed to it from SMRT. The sensor passband is defined using the centre frequency, the intermediate frequency (i.e. the offset (±) of the passband centres from the channel centre), and the bandwidth. These are passed using the `channel` keyword, e.g.
```python
# Create realistic atmosphere for sensor channel at 183+/-3 GHz with 1GHz bandwidth
atm = RealisticAtmosphere("/path/to/profile.xml", 300.0, 0.0,
                          channel = [183e9, 3e9, 1e9])
```

The default settings for the atmospheric absorption calculations is to use the ``AER v3.6`` line parameters for water vapour with the ``MTCKD v3.2`` continuum, and the ``Tretyakov 2005`` model for oxygen. These can be altered to use any of the _complete absorption models_ for water vapour and oxygen available in ARTS (see ARTS user guide for details) using the ``water_vapour_model`` and ``oxygen_model`` keywords. For example, to use the `Rosenkranz 1998` models for both oxygen and water vapour:
```python
atm = RealisticAtmosphere("/path/to/profile.xml", 300.0, 0.0,
                          water_vapour_model="PWR98",
                          oxygen_model="PWR98")
```