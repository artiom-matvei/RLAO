# -*- coding: utf-8 -*-
"""
OOPAO module for the Reinforcement learning structure (also used for the integrator)
@author: Raissa Camelo (LAM) git: @srtacamelo
@author: cheritier (OOPAO Pyramid_WFS closed loop tutorial )
"""

from .__load__oopao import load_oopao
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import os

class OOPAO(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    #--------------------------Core gym funtions--------------------------------
    def __init__(self):

        # OOPAO Modules
        self.gainCL = None
        self.atm = None
        self.wfs = None
        self.dm = None
        self.misReg = None
        self.tel = None
        self.src = None
        self.ngs = None
        self.imat = None          # zonal imat
        self.M2C_CL = None
        self.calib_CL = None
        self.mode = None
        self.Z = None
        self.plot_obj = None
        self.display = None
        self.cam = None
        self.cam_binned = None
        self.reconstructor = None
        # OOPAO saved info
        self.SE_PSF = None
        self.LE_PSF = None
        self.LE_PSFs = None
        self.SR        = None
        self.total     = None
        self.residual  = None
        self.wfsSignal = None
        self.OPD = None

        # Jalo
        self.action_buffer = []
        self.done = False
        self.param_file = ""
        self.oopao_path = ""
        self.delay = 1
        self.S2V = None
        self.V2S = None
        self.F = 1
        self.pmat = None
        self.infmat = None
        self.calibConst = 1
        self.name = "OOPAO"

        self.dm_mask = None
        self.nActuator = None
        self.xvalid = None
        self.yvalid = None

        self.leak = 0.99
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None


    def reset(self):
        self.set_params()
        self.action_buffer = []
        obs = -np.matmul(self.reconstructor,self.get_slopes())
        obs = self.vec_to_img(obs)*1e6
        return obs
    
    def reset_soft(self):
        self.action_buffer = []
        obs = -np.matmul(self.reconstructor,self.get_slopes())
        obs = self.vec_to_img(obs)*1e6
        return obs
    # ---------------- Simulation initialisation functions ---------------------
    def set_params_file(self,param_file,oopao_path):
        if param_file != self.param_file:
            self.param_file = param_file
        if oopao_path != self.oopao_path:
            self.oopao_path = oopao_path
    def set_params(self, args,wfs_type = "pyramid", modal_basis = "zernike",gainCL = 0.5):

        self.gainCL = gainCL
        load_oopao(self.oopao_path) # 

        import importlib

        # The file gets executed upon import, as expected.
        config = importlib.import_module(self.param_file)

        param = config.initializeParameterFile(args)

        self.param = param
        
        import time
        from OOPAO.Atmosphere import Atmosphere
        from OOPAO.DeformableMirror import DeformableMirror
        from OOPAO.MisRegistration import MisRegistration
        from OOPAO.Pyramid import Pyramid
        from OOPAO.Source import Source
        from OOPAO.Telescope import Telescope
        from OOPAO.Zernike import Zernike
        from OOPAO.calibration.CalibrationVault import CalibrationVault
        from OOPAO.calibration.InteractionMatrix import InteractionMatrix
        from OOPAO.tools.displayTools import cl_plot, displayMap



        #%% -----------------------     TELESCOPE   ----------------------------------

        # create the Telescope object
        self.tel = Telescope(resolution           = param['resolution'],                      # resolution of the telescope in [pix]
                             diameter             = param['diameter'],                        # diameter in [m]        
                             samplingTime         = param['samplingTime'],                    # Sampling time in [s] of the AO loop
                             centralObstruction   = param['centralObstruction'],              # Central obstruction in [%] of a diameter 
                             display_optical_path = False,                                    # Flag to display optical path
                             fov                  = 1 )      

        #%% -----------------------     NGS   ----------------------------------
        # create the Natural Guide Star object
        self.ngs = Source(optBand     = param['opticalBand'],        # Optical band (see photometry.py)
                          magnitude   = param['magnitude'],        # Source Magnitude
                          coordinates = [0,0])                       # Source coordinated [arcsec,deg]

        # combine the NGS to the telescope using '*'
        self.ngs*self.tel

        # create the Scientific Target object located at 10 arcsec from the  ngs
        self.src = Source(optBand     = param['opticalBand'],           # Optical band (see photometry.py)
                          magnitude   = param['magnitude'],           # Source Magnitude
                          coordinates = [1,0])                          # Source coordinated [arcsec,deg]

        # combine the SRC to the telescope using '*'
        self.src*self.tel

        # check that the ngs and tel.src objects are the same
        self.tel.src.print_properties()

        # compute PSF 
        self.tel.computePSF(zeroPaddingFactor = 6)

        #%% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        self.atm=Atmosphere(telescope     = self.tel,\
                            r0            = param['r0'],\
                            L0            = param['L0'],\
                            windSpeed     = param['windSpeed'],\
                            fractionalR0  = param['fractionnalR0'],\
                            windDirection = param['windDirection'],\
                            altitude      = param['altitude'])
        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)

        # The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
        self.atm.update()

        # display the atmosphere layers for the sources specified in list_src: 
        # self.atm.display_atm_layers(list_src=[self.ngs,self.src])

        # the sources coordinates can be updated on the fly: 
        self.src.coordinates = [0,0]
        # self.atm.display_atm_layers(list_src=[self.ngs,self.src])

        
        #%% -----------------------     Scientific Detector   ----------------------------------
        from OOPAO.Detector import Detector

        # define a detector with its properties (see Detector class for further documentation)
        self.cam = Detector(integrationTime = self.tel.samplingTime,      # integration time of the detector
                       photonNoise     = True,                  # enable photon noise
                       readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
                       QE              = 1,                   # quantum efficiency
                       psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                       binning         = 1)                     # Binning factor of the PSF

        self.cam_binned = Detector( integrationTime = self.tel.samplingTime,      # integration time of the detector
                               photonNoise     = True,                  # enable photon noise
                               readoutNoise    = 2,                     # readout of the detector in [e-/pixel]
                               QE              = 0.8,                   # quantum efficiency
                               psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                               binning         = 4)                     # Binning factor of the PSF


        # computation of a PSF on the detector using the '*' operator
        self.src*self.tel*self.cam*self.cam_binned

        
        #%%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
        # The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere): 
        self.tel+self.atm # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated. 


        # computation of a PSF on the detector using the '*' operator
        self.atm*self.ngs*self.tel*self.cam*self.cam_binned

        # The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation) 
        self.tel-self.atm

        #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration()
        misReg.shiftX = param['MisReg_shiftX']                  # in [m]
        misReg.shiftY = param['MisReg_shiftY']                  # in [m]
        misReg.rotationAngle = param['MisReg_rotationAngle']    # in [deg]

        # Get valid Actuators
        self.nActuator = param['nActuator']

        # if no coordonates specified, create a cartesian dm
        self.dm=DeformableMirror(telescope    = self.tel,                     # Telescope
                                 nSubap       = param['nSubaperture'],        # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)                   
                                 mechCoupling = param['mechanicalCoupling'],  # Mechanical Coupling for the influence functions
                                 misReg       = misReg,                       # Mis-registration associated
                                 coordinates  = None,                         # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                                 pitch        = self.tel.D/self.nActuator)    # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
        
        # Get Valid Actuators mask (X and Y axis) this si important for the RL      
        self.dm_mask = np.reshape(self.dm.validAct,(self.nActuator,self.nActuator))
        (self.xvalid, self.yvalid) = np.nonzero(self.dm_mask)
        #%% -----------------------     PYRAMID WFS   ----------------------------------

        # make sure tel and atm are separated to initialize the PWFS
        self.tel.isPaired = False
        self.tel.resetOPD()

        self.wfs = Pyramid(nSubap       = param['nSubaperture'],        # number of subaperture = number of pixel accros the pupil diameter       
                      telescope         = self.tel,                     # telescope object
                      lightRatio        = param['lightThreshold'],      # flux threshold to select valid sub-subaperture
                      modulation        = param['modulation'],          # Tip tilt modulation radius
                      binning           = 1,                            # binning factor (applied only on the )
                      n_pix_separation  = param['n_pix_separation'],    # number of pixel separating the different pupils
                      n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
                      postProcessing    = param['postProcessing'])  # slopesMaps,

        # Propagate the light to the Wave-Front Sensor
        self.tel*self.wfs
        #%% -----------------------     Modal Basis - Zernike  ----------------------------------
        #%% ZERNIKE Polynomials
        # create Zernike Object
        Z = Zernike(self.tel,50)
        # compute polynomials for given telescope
        Z.computeZernike(self.tel)

        # mode to command matrix to project Zernike Polynomials on DM
        # M2C_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical,:]))@Z.modes

        M2C_zernike = np.load(os.path.dirname(__file__)+'/manual_m2c.npy')[:, :50]

        # show the first 10 zernikes
        self.dm.coefs = M2C_zernike[:,:10]
        self.tel*self.dm
        displayMap(self.tel.OPD)

        #%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

        # amplitude of the modes in m
        stroke=self.ngs.wavelength/16
        # zonal Interaction Matrix
        M2C_zonal = np.eye(self.dm.nValidAct)
        # modal Interaction Matrix for 300 modes
        M2C_modal = M2C_zernike[:,:300]

        self.tel-self.atm
        # zonal interaction matrix
        calib_modal = InteractionMatrix(ngs            = self.ngs,
                                        atm            = self.atm,
                                        tel            = self.tel,
                                        dm             = self.dm,
                                        wfs            = self.wfs,   
                                        M2C            = M2C_zonal, # M2C matrix used 
                                        stroke         = stroke,    # stroke for the push/pull in M2C units
                                        nMeasurements  = 6,        # number of simultaneous measurements
                                        noise          = 'off',     # disable wfs.cam noise 
                                        display        = True,      # display the time using tqdm
                                        single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


    


        # Modal interaction matrix
        calib_zernike = CalibrationVault(calib_modal.D@M2C_zernike)
        #%%
        #%%  ----------------------- Define instrument and WFS path detectors  -----------------------

        # instrument path
        src_cam = Detector(self.tel.resolution*4)
        src_cam.psf_sampling = 4
        src_cam.integrationTime = self.tel.samplingTime*1
        # put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
        self.src.coordinates = [0.4,0]

        # WFS path
        ngs_cam = Detector(self.tel.resolution)
        ngs_cam.psf_sampling = 4
        ngs_cam.integrationTime = self.tel.samplingTime

        # initialize DM commands
        self.tel.resetOPD()
        self.dm.coefs=0
        self.dm_prev = self.dm.coefs.copy()
        self.ngs*self.tel*self.dm*self.wfs
        self.wfs*self.wfs.focal_plane_camera

        # Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
        # atm.r0 = 0.15
        self.atm.generateNewPhaseScreen(seed = 10)

        self.tel+self.atm


        self.tel.computePSF(4)
        
        # These are the calibration data used to close the loop
        calib_CL    = calib_zernike
        M2C_CL      = M2C_zernike


        # combine telescope with atmosphere
        self.tel+self.atm
        # initialize DM commands
        self.atm*self.ngs*self.tel*ngs_cam
        self.atm*self.src*self.tel*src_cam
        
        # allocate memory to save data
        self.SR                      = []   #np.zeros(param['nLoop'])
        self.total                   = np.zeros(param['nLoop'])
        self.residual                = np.zeros(param['nLoop'])
        self.wfsSignal               = np.arange(0,self.wfs.nSignal)*0
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.LE_PSFs = []
        
        # from OOPAO.tools.displayTools import cl_plot

        # self.plot_obj = cl_plot(list_fig  = [self.atm.OPD,
        #                                 self.tel.mean_removed_OPD,
        #                                 self.tel.mean_removed_OPD,
        #                                 [[0,0],[0,0],[0,0]],
        #                                 self.wfs.cam.frame,
        #                                 self.wfs.focal_plane_camera.frame,
        #                                 np.log10(self.tel.PSF),
        #                                 np.log10(self.tel.PSF)],
        #                     type_fig          = ['imshow',
        #                                          'imshow',
        #                                          'imshow',
        #                                          'plot',
        #                                          'imshow',
        #                                          'imshow',
        #                                          'imshow',
        #                                          'imshow'],
        #                     list_title        = ['Turbulence [nm]',
        #                                          'NGS@'+str(self.ngs.coordinates[0])+'" WFE [nm]',
        #                                          'SRC@'+str(self.src.coordinates[0])+'" WFE [nm]',
        #                                          None,
        #                                          'WFS Detector',
        #                                          'WFS Focal Plane Camera',
        #                                          None,
        #                                          None],
        #                     list_legend       = [None,None,None,['SRC@'+str(self.src.coordinates[0])+'"','NGS@'+str(self.ngs.coordinates[0])+'"'],None,None,None,None],
        #                     list_label        = [None,None,None,['Time','WFE [nm]'],None,None,['NGS PSF@'+str(self.ngs.coordinates[0])+'" -- FOV: '+str(np.round(ngs_cam.fov_arcsec,2)) +'"',''],['SRC PSF@'+str(self.src.coordinates[0])+'" -- FOV: '+str(np.round(src_cam.fov_arcsec,2)) +'"','']],
        #                     n_subplot         = [4,2],
        #                     list_display_axis = [None,None,None,True,None,None,None,None],
        #                     list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)

        self.wfs.cam.photonNoise     = True
        self.display                 = True
        self.reconstructor = M2C_CL@calib_CL.M
        self.F = M2C_CL @ np.linalg.pinv(M2C_CL)

        self.M2C_CL = M2C_CL


    def set_wfs(self,param,type = "pyramid"):
        if type == "pyramid":

            from OOPAO.Pyramid          import Pyramid
            # make sure tel and atm are separated to initialize the PWFS
            self.tel-self.atm
            
            self.wfs = Pyramid(nSubap              = param['nSubaperture'],\
                        telescope             = self.tel,\
                        modulation            = param['modulation'],\
                        lightRatio            = param['lightThreshold'],\
                        n_pix_separation      = param['n_pix_separation'],\
                        psfCentering          = param['psfCentering'],\
                        postProcessing        = param['postProcessing'])

            self.tel*self.wfs

        

    def set_modalBasis(self, mode = "zernike"):
        if mode == "zernike":
            from OOPAO.Zernike import Zernike
            from OOPAO.calibration.CalibrationVault import CalibrationVault
            from OOPAO.calibration.InteractionMatrix import InteractionMatrix

            # create Zernike Object
            Z = Zernike(self.tel,50)
            # compute polynomials for given telescope
            Z.computeZernike(self.tel)

            # mode to command matrix to project Zernike Polynomials on DM
            M2C_zernike  = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical,:]))@Z.modes

            #self.dm.coefs = M2C_zernike[:,:10]
            #self.tel*self.dm

            M2C_zonal = np.eye(self.dm.nValidAct)
            # zonal interaction matrix
            self.imat = InteractionMatrix(  ngs      = self.source,\
                                            atm            = self.atm,\
                                            tel            = self.tel,\
                                            dm             = self.dm,\
                                            wfs            = self.wfs,\
                                            M2C            = M2C_zonal,\
                                            stroke         = 1e-9,\
                                            nMeasurements  = 25,\
                                            noise          = 'off')
            # Modal interaction matrix
            calib_zernike = CalibrationVault(self.imat.D@M2C_zernike)

            self.M2C_CL = M2C_zernike
            self.calib_CL = calib_zernike

    def set_display(self):
        from OOPAO.tools.displayTools import cl_plot
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.plot_obj = cl_plot(list_fig  = [self.atm.OPD,self.tel.mean_removed_OPD,self.wfs.cam.frame,np.log10(self.wfs.get_modulation_frame(radius = 1)),[[0,0],[0,0]],[self.dm.coordinates[:,0],np.flip(self.dm.coordinates[:,1]),self.dm.coefs],np.log10(self.tel.PSF),np.log10(self.tel.PSF)],\
                   type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD','Residual OPD','WFS Detector','WFS Modulation Camera',None,None,None,None],\
                   list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-4,0]],\
                   list_label        = [None,None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
    def render(self, current_i,mode='rgb_array'):
        """
        Render and display the images of the WFS in real time.
        """
        from OOPAO.tools.displayTools import cl_plot
        # update displays if required
        if self.display==True:        
            self.tel.computePSF(4)
            if current_i>15:
                self.SE_PSF.append(np.log10(self.tel.PSF))
                self.LE_PSF = np.mean(self.SE_PSF, axis=0)
            
            plot = cl_plot(list_fig   = [self.atm.OPD,self.tel.mean_removed_OPD,self.wfs.cam.frame,
                                np.log10(self.wfs.get_modulation_frame(radius=10)),
                                [np.arange(current_i+1),self.residual[:current_i+1]],
                                self.dm.coefs,np.log10(self.tel.PSF), self.LE_PSF],
                                plt_obj = self.plot_obj)
                                
            plt.pause(0.1)
        return self.LE_PSF, np.log10(self.tel.PSF)
    def render4plot(self,current_i):
        self.tel.computePSF(4)
        if current_i>15:
            self.SE_PSF.append(np.log10(self.tel.PSF.copy()))
            self.LE_PSF = np.mean(self.SE_PSF, axis=0)
        #plt.imshow(self.LE_PSF)
        #plt.show()
        #plt.pause(3)
        #plt.close()
        return self.LE_PSF,np.log10(self.tel.PSF)


    def step(self,i, action ): # action, showAtmos = True
        # Single AO step. Action defines DM shape and function returns WFS
        # slopes, reward (-1*norm of slopes), done as false (no current condition)
        # and (currently empty) info dictionary, where one could store useful
        # data about the simulation

        action = self.img_to_vec(action)*1e-6

        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        self.atm.update()

        # save phase variance
        self.total[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9

        # save turbulent phase
        turbPhase = self.tel.src.phase
        # propagate to the WFS with the CL commands applied
        
        self.tel*self.dm*self.wfs

        # Integrator
        # self.dm.coefs=self.dm.coefs-self.gainCL*np.matmul(self.reconstructor,self.wfsSignal)  
        
        self.dm.coefs = (self.dm_prev * self.leak) + action 
        self.dm_prev = self.dm.coefs.copy()
    

        # store the slopes after computing the commands => 2 frames delay
        
        self.wfsSignal=self.wfs.signal
        
        slopes = self.wfsSignal 
        obs = -np.matmul(self.reconstructor,slopes)
        obs = self.vec_to_img(obs)*1e6
        

        
        self.residual[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        self.OPD=self.tel.OPD[np.where(self.tel.pupil>0)]
      
        # Save Strehl ratio to calculate Episode Average
        
        strehl = self.get_strehl()
        self.SR.append(strehl)
       
        # Extra
        info = {"strehl":strehl}
        done = 0

        wfsf = self.wfs.cam.frame

        return obs, wfsf, -1 * np.linalg.norm(obs), strehl,bool(done), info
    
    def calculate_strehl_AVG(self):
        """Calculates the average strehl ratio for each episode.
        Cleans the strehl array after each episode.
        """
        avg = np.mean(self.SR)
        std = np.std(self.SR)
        self.SR = []

        return avg, std
    
    def integrator(self):
        return -self.gainCL*np.matmul(self.reconstructor,self.wfsSignal)  
        
    def get_slopes(self):
        return self.wfs.signal
    
    def get_strehl(self):
        return  np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil==1)]))

    def _get_reward(self,slopes,type = "volt"):
        if self.S2V is not None and type is not "sh":
            res_volt = np.matmul(self.S2V, slopes)
            reward = -1 * np.linalg.norm(res_volt)
        else:
            reward = self.get_strehl()

        return reward
    
    def sample_noise(self, sigma):

        noise = self.F @ (sigma * np.random.normal(0,1 , size = (int(self.dm.nValidAct),))) # 357
        return self.vec_to_img(noise)
    
    def vec_to_img(self, action_vec, use_torch=False):
        if use_torch:
            valid_actus = torch.zeros((self.nActuator, self.nActuator)).to(self.device)

        else:
            valid_actus = np.zeros((self.nActuator, self.nActuator))

        valid_actus[self.xvalid, self.yvalid] = action_vec

        return valid_actus

    def img_to_vec(self, action):
        # assert len(action.shape) == 2
        if len(action.shape) == 4:
            action_out = action[:,:,self.xvalid, self.yvalid]
        else:
            action_out = action[self.xvalid, self.yvalid]
        
        return action_out

        
    def load_network(self, path, model):

        checkpoint = torch.load(path ,map_location=self.device)

        # Make sure to use the correct network before loading the state dict
        self.network = model(self.xvalid,self.yvalid)
        # Restore the regular model and optimizer state
        self.network.load_state_dict(checkpoint['model_state_dict'])

        self.network.to(self.device)

        self.network.eval()