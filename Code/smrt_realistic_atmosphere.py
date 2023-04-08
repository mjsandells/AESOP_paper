'''Atmosphere using ARTS to return downwelling, upwelling and transmission at
multiple viewing angles.'''

import pathlib
import numpy as np
from pyarts.workspace import Workspace, arts_agenda


class RealisticAtmosphere():
    '''Representation of a realistic atmosphere using ARTS for a given atmospheric profile'''
    def __init__(self, profile, sensor_height, ground_height, channel=None,
                 water_vapour_model=None, oxygen_model='TRE05',
                 line_file='aer_3.6_lines_0_5THz.H2O.xml.gz', arts_verbosity=0):
        '''
        Args
            profile (filename or array): atmospheric profile, either ARTS xml or array. Columns
                are pressure (Pa), temperature (K), altitude (m), water vapour content (vmr).
            sensor_height (float): height of sensor (aircraft) to get upwelling at aircraft height.
            ground_height (float): height at base of profile (ground) used to get downwelling at
                surface.
            channel (array): optional channel definition for use with dual-sideband instruments
                defined using (centre frequency, IF frequency, IF bandwidth)
                e.g. ch_89 = pd.DataFrame((88.992e9, 1.075e9,  650e6), columns=['M16-89']).T
            water_vapour_model (string): ARTS absorption model for water vapour. See ARTS
                documentation on complete absorption models for options.
            oxygen_model (string): ARTS absorption model for oxygen. See ARTS documentation on
                complete absorption models for options.
            arts_verbosity (int): Verbosity level for ARTS output
        '''
        self.channel = channel

        if isinstance(profile, np.ndarray):
            self.profile = profile
        else:
            import pyarts.xml as axml
            self.profile = axml.load(profile)
        self.sensor_height = sensor_height
        # Set ground height to maximum of requested value and
        # the lowest height in the input profile
        self.ground_height = max(ground_height, self.profile[0, 2])
        self.water_vapour_model = water_vapour_model
        self.oxygen_model = oxygen_model
        self.line_file = line_file
        self.arts_verbosity = arts_verbosity

    def tb_and_trans(self, frequency, costheta, output):
        '''Return array of downwelling brightness temperatures (TRM - thermodynamic brightness
        temperature) at multiple view angles.

        Args
            frequency: Monochromatic frequency (if no channel definition created in
                realistic_atmosphere object).
            costheta (array or float): array (or float) of cosines which are converted to view
                angles in degrees.
            output: 'tbup' for upwelling brightness temperature, 'tbdown' for downwelling brightness
                temperature, 'trans' for transmission

        Returns
            Array of downwelling brightness temperatures at multiple view angles.
        '''
            
        channel = self.channel    
               
        if isinstance(channel, dict):
            return {channel[k][0]:self.arts_calc(output, costheta, frequency, np.atleast_2d(channel[k])).copy() for k in channel}
        elif channel is not None:
            return self.arts_calc(output, costheta, frequency, np.atleast_2d(channel))
        else:
            return self.arts_calc(output, costheta, frequency, channel).copy()
        

    def tbdown(self, frequency, costheta, npol):
        '''Return downwelling brightness temperature'''
        tbdown = self.tb_and_trans(frequency, costheta, 'tbdown')
        if isinstance(tbdown, dict):
            return(np.stack((tbdown[frequency], )*npol, axis=1)).flatten()
        else:
            return(np.stack((tbdown, )*npol, axis=1)).flatten()

    def tbup(self, frequency, costheta, npol):
        '''Return upwelling brightness temperature'''
        tbup = self.tb_and_trans(frequency, costheta, 'tbup')
        if isinstance(tbup, dict):
            return(np.stack((tbup[frequency], )*npol, axis=1)).flatten()
        else:
            return(np.stack((tbup, )*npol, axis=1)).flatten()

    def trans(self, frequency, costheta, npol):
        '''Return transmission between surface and sensor'''
        trans = self.tb_and_trans(frequency, costheta, 'trans')
        if isinstance(trans, dict):
            return(np.stack((trans[frequency], )*npol, axis=1)).flatten()
        else:
            return(np.stack((trans, )*npol, axis=1)).flatten()
    
    def arts_calc(self, output, costheta, frequency=None, channel=None):
        self._setup_arts(channel)
        if channel is None:
            if frequency is None:
                print ('Either frequency or channel must be specified')
            self.workspace.f_grid = np.atleast_1d(frequency)
            self.workspace.sensorOff()

        costheta = np.atleast_1d(np.array(costheta))
        angles = np.degrees(np.arccos(costheta))
        if (output == 'tbup') or (output == 'trans'):
            angles = 180 - angles
        self.workspace.sensor_los = angles[:, np.newaxis]
        if output == 'tbdown':
            sim_height = self.ground_height
        else:
            sim_height = self.sensor_height
        self.workspace.sensor_pos = sim_height * np.ones_like(self.workspace.sensor_los.value)
        
        
        self.workspace.sensor_checkedCalc()
        self.workspace.yCalc()
        
        if (output == 'tbup') or (output == 'tbdown'):
            return self.workspace.y.value
        elif output == 'trans':
            return np.exp(-1.0*self.workspace.y_aux.value[0])
        else:
            raise Exception(f'Unknown output type {output}')
        
    def _setup_arts(self, channel):
        '''Set up ARTS to perform atmospheric radiative transfer'''
        self.workspace = Workspace(verbosity=self.arts_verbosity)

        # General settings
        self.workspace.execute_controlfile('general/general.arts')
        self.workspace.execute_controlfile('general/continua.arts')
        self.workspace.execute_controlfile('general/agendas.arts')
        self.workspace.execute_controlfile('general/planet_earth.arts')
        self.workspace.Copy(self.workspace.abs_xsec_agenda,
                            self.workspace.abs_xsec_agenda__noCIA)
        self.workspace.Copy(self.workspace.iy_main_agenda,
                            self.workspace.iy_main_agenda__Emission)
        self.workspace.Copy(self.workspace.iy_surface_agenda,
                            self.workspace.iy_surface_agenda__UseSurfaceRtprop)
        self.workspace.Copy(self.workspace.ppath_agenda,
                            self.workspace.ppath_agenda__FollowSensorLosPath)
        self.workspace.Copy(self.workspace.ppath_step_agenda,
                            self.workspace.ppath_step_agenda__GeometricPath)
        self.workspace.Copy(self.workspace.iy_space_agenda,
                            self.workspace.iy_space_agenda__CosmicBackground)
        self.workspace.Copy(self.workspace.propmat_clearsky_agenda,
                            self.workspace.propmat_clearsky_agenda__OnTheFly)
        self.workspace.iy_unit = 'PlanckBT'
        self.workspace.stokes_dim = 1
        self.workspace.atmosphere_dim = 1
        self.workspace.iy_aux_vars = ['Optical depth']

        # Gas absorption settings
        if self.water_vapour_model is not None: # Complete absorption model specified for H2O
            self.workspace.abs_speciesSet(species=[f'H2O-{self.water_vapour_model}',
                                                   f'O2-{self.oxygen_model}',
                                                   'N2-SelfContStandardType'])
            self.workspace.abs_lines_per_speciesSetEmpty()
        else: # Default H2O setup using AER lines and MT-CKD 3.2 continuum
            self.workspace.abs_speciesSet(species=['H2O,H2O-SelfContCKDMT320,'
                                                   'H2O-ForeignContCKDMT320',
                                                   f'O2-{self.oxygen_model}',
                                                   'N2-SelfContStandardType'])
            line_file = self.line_file
            self.workspace.ReadARTSCAT(filename=line_file,
                                       normalization_option='VVH',
                                       mirroring_option='Same',
                                       lineshapetype_option="VP",
                                       cutoff_option="ByLine",
                                       cutoff_value=750e9)
            self.workspace.abs_lines_per_speciesCreateFromLines()

        self.workspace.atm_fields_compactFromMatrix(gin1=self.profile,
                                                    field_names=['T', 'z',
                                                                 'abs_species-H2O'])
        self.workspace.atm_fields_compactAddConstant(name='abs_species-O2', value=0.2095,
                                                     condensibles=['abs_species-H2O'])
        self.workspace.atm_fields_compactAddConstant(name='abs_species-N2', value=0.7808,
                                                     condensibles=['abs_species-H2O'])
        self.workspace.AtmFieldsAndParticleBulkPropFieldFromCompact()
        self.workspace.lbl_checkedCalc()

        # Surface settings
        @arts_agenda
        def no_surface(workspace):
            workspace.Copy(workspace.surface_skin_t, 0.1)
            workspace.surfaceBlackbody()
        self.workspace.Copy(self.workspace.surface_rtprop_agenda, no_surface)

        self.workspace.z_surface = np.atleast_2d(max(self.ground_height,
                                                     self.workspace.z_field.value[0]))

        # Sensor settings
        if channel is not None:
            self.workspace.sensor_description_amsu = channel
            self.workspace.sensor_responseSimpleAMSU()

        # Jacobian settings
        self.workspace.jacobianOff()

        # Cloud settings
        self.workspace.cloudboxOff()

        # Checks that can be done now
        self.workspace.atmfields_checkedCalc(bad_partition_functions_ok=1)
        self.workspace.abs_xsec_agenda_checkedCalc()
        self.workspace.atmgeom_checkedCalc()
        self.workspace.cloudbox_checkedCalc()
        self.workspace.propmat_clearsky_agenda_checkedCalc()