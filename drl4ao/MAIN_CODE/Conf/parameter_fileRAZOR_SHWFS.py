# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

@author: cheritie - astriffl
"""
import numpy as np
from OOPAO.tools.tools import createFolder

def initializeParameterFile(args):
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = args.r0                                           # value of r0 in the visibile in [m]
    param['L0'                   ] = args.L0                                            # value of L0 in the visibile in [m]
    param['fractionalR0'         ] = args.fractionalR0                                            # Cn2 profile
    param['windSpeed'            ] = args.windSpeed                                           # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = args.windDirection                                            # wind direction of the different layers in [degrees]
    param['altitude'             ] = args.altitude                                        # altitude of the different layers in [m]
                      
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 1                                      # diameter in [m]
    param['nSubaperture'         ] = 22                                            # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 16                                            # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap'] # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']       # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/500                                         # loop sampling time in [s]
    param['centralObstruction'   ] = 0                                            # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                             # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 0.01                                             # reflectivity of the 798 segments
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 3.                                              # magnitude of the guide star
    param['opticalBand'          ] = 'R'                                            # optical band of the guide star                                          # optical band of calibration laser
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = 11                                             # number of actuators 
    param['mechanicalCoupling'   ] = 0.4
    param['isM4'                 ] = False                                           # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the eformable mirror class
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0#-0.18                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0#-4.83                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0#-0.37                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0#-0.19   
    


    mask = np.array([[0,0,0,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,1,1,1,0],
          [1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,1,1,0],
          [0,0,1,1,1,1,1,1,1,0,0],[0,0,0,1,1,1,1,1,0,0,0]], dtype=int)   
    
    param['boolActMask'] = mask.astype(bool)
    
    mask[mask == 0] = 1e10

    param['actMask'] = mask

    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['modulation'            ] = 6                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 10                                            # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'slopesMaps'                         # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'



    im = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,\
        1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,\
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,\
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, \
        1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, \
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,\
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
        1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,\
        1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,\
        0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
    param['validMask'              ] = im.reshape((22,22)).astype(bool)

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = args.nLoop                                         # number of iteration                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = 0.5                                          # integrator gain
    param['nModes'                ] = 97                                          # number of KL modes controlled 
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector to get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'RAZOR_mod_3'

    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param

