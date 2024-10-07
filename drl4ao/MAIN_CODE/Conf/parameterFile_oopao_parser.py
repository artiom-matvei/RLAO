# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:02 2020

@author: cheritie
"""

import numpy as np
from os import path
from OOPAO.tools.tools  import createFolder


def initializeParameterFile(args):
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = args.r0                                           # value of r0 in the visibile in [m]
    param['L0'                   ] = args.L0                                           # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = args.fractionnalR0                                # Cn2 profile
    param['windSpeed'            ] = args.windSpeed                                    # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = args.windDirection                                # wind direction of the different layers in [degrees]
    param['altitude'             ] = args.altitude                                     # altitude of the different layers in [m]
                    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 1.52                                               # diameter in [m]
    param['nSubaperture'         ] = 20 #40                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 6                                                  # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']      # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']            # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/500                                              # loop sampling time in [s]
    param['centralObstruction'   ] = 0                                                  # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                                  # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 1                                                  # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 8                                              # magnitude of the guide star 0
    param['opticalBand'          ] = 'I'                                            # optical band of the guide star
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = param['nSubaperture']+1   #17                  # number of actuators 
    param['mechanicalCoupling'   ] = 0.35
    param['isM4'                 ] = True                                           # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the eformable mirror class
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                                              # tangential scaling in percentage of diameter
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MISREGISTRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['MisReg_shiftX'] = 0                                                      # in [m]
    param['MisReg_shiftY'] = 0                                                      # in [m]
    param['MisReg_rotationAngle'] = 0                                               # in [deg]
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = args.modulation                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 4                                             # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'slopesMaps_incidence_flux'                                   # post-processing of the PWFS signals 
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = args.nLoop                                       # number of iterations                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = args.gainCL                                  # Integrator gain
    param['nModes'                ] = int(15*15*(np.pi/4))   #   600               # number of KL modes controlled 
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector too get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Un-comment if needed
    """
    # name of the system
    param['name'] = 'T152_PAPYRUS' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    """
    return param
