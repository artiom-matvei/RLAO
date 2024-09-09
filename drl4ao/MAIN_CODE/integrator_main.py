# -*- coding: utf-8 -*-
"""
OOPAO module for the integrator
@author: Raissa Camelo (LAM) git: @srtacamelo
"""
from Conf.parser_Configurations import Config, ConfigAction
from OOPAOEnv.OOPAOEnv import OOPAO
from PO4AO.util_simple import TorchWrapper
from Plots.plots import save_plots
import argparse
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_env(args,gainCL=0.9):
    env = OOPAO()
    env.set_params_file(args.param_file,args.oopao_path) # 
    env.set_params(args,"pyramid",gainCL = gainCL)   #sets env parameter file
    return TorchWrapper(env)

def get_1ListFromIntStr(paramStr):
    mylist = []
    for i in paramStr.split(','):
        try:
            mylist.append(int(i))

        except ValueError:
            mylist.append(float(i))
    return(mylist)

def commandLine():
    # CREATE ArgumentParser OBJECT :
    parser = argparse.ArgumentParser(description='TRAINING STEP for a REGRESSOR'
                                     ' by using a NEURAL NETWORK',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data Preferences
    parser.add_argument('--save_all_PSF', help='Save all Long and Short exposure PSF.',
                        default=False, type=bool, action=ConfigAction)
    parser.add_argument('--savedir', help='Neme of the folder in the log folder.',
                        default='../../../logs/AO4ELT7', type=str, action=ConfigAction)
    # OOPAO Simulation Parameters
    # MBRL Parameters
    # OOPAO Simulation Parameters
    parser.add_argument('--experiment_tag', help="Tag to identify the experiment's result folder",
                        default="standard", type=str, action=ConfigAction)
    parser.add_argument('--oopao_path', help='OOPAO lib path',
                        default="AO_OOPAO", type=str, action=ConfigAction)
    parser.add_argument('--param_file', help='OOPAO Simulation parameter file',
                        default="Conf.parameterFile_oopao_parser", type=str, action=ConfigAction)
    #------- OOPAO ATMOSPHERE --------
    parser.add_argument('--r0', help='Value of r0 in the visibile in [m]',
                        default=0.13, type=float, action=ConfigAction)
    parser.add_argument('--L0', help='Value of L0 in the visibile in [m]',
                        default=30, type=float, action=ConfigAction)
    parser.add_argument('--fractionnalR0', help='Cn2 profile',
                        default="0.45,0.1,0.1,0.25,0.1" ,type=str, action=ConfigAction)
    parser.add_argument('--windSpeed', help='Wind speed of the different layers in [m.s-1]',
                        default="10,12,11,15,20",type=str, action=ConfigAction)
    parser.add_argument('--windDirection', help='Wind direction of the different layers in [degrees]',
                        default="0,72,144,216,288",type=str, action=ConfigAction)
    parser.add_argument('--altitude', help='Altitude of the different layers in [m]',
                        default="0,1000,5000,10000,12000", nargs='+',type=str, action=ConfigAction)
    #------- OOPAO LOOP PROPERTIES--------
    parser.add_argument('--nLoop', help='Number of iterations (integrator simulations)',
                        default=10000, type=int, action=ConfigAction)
    parser.add_argument('--gainCL', help='Integrator gain',
                        default=0.5 , type=int, action=ConfigAction)
    parser.add_argument('--gain_list', help='List of optical gains for the integrator to simulate',
                        default="0.1, 0.3,0.5, 0.7,0.9,0.95,1,1.1,1.2,1.5,1.7,2.0,2.5", type=str, action=ConfigAction)
    parser.add_argument('--frames_per_sec', help='Number of frames per Episode/ Second',
                        default=500, type=int, action=ConfigAction)
    # CONFIGURATION FILE :
    parser.add_argument('--config', help='Parameters .json file', type=str)
    args = parser.parse_args()
    return(Config(args))

if __name__=='__main__':
    #Load Parameters
    args = commandLine()
    args.windSpeed = get_1ListFromIntStr(args.windSpeed)
    args.windDirection = get_1ListFromIntStr(args.windDirection)
    args.altitude = get_1ListFromIntStr(args.altitude)
    args.fractionnalR0 = get_1ListFromIntStr(args.fractionnalR0)
    args.gain_list = get_1ListFromIntStr(args.gain_list)



    for gainCL in args.gain_list: # Loop through all the simulations with different integrator gains per job 
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        savedir = args.savedir+'/integrator/'+f'{timestamp}'+'_'+args.experiment_tag+'_'+str(int(args.nLoop/args.frames_per_sec))+'s'+"_"+str(gainCL)
        

        os.makedirs(savedir, exist_ok=True)
        args.save(savedir+"/arguments.json")

        env = get_env(args,gainCL = gainCL)

        print("Running loop...")

        
        SRs = []
        rewards = []
        accu_reward = 0

        obs = env.reset_soft()

        
        for i in range(args.nLoop): # Loop through the number of frames per simulation
            a=time.time()
            action = env.gainCL * obs #env.integrator()
            obs, reward,strehl, done, info = env.step(i,action)  
            accu_reward+= reward
            

            b= time.time()
            print('Elapsed time: ' + str(b-a) +' s')
            
            print('Loop '+str(i+1)+'/'+str(args.nLoop)+' Gain: '+str(gainCL)+' Turbulence: '+str(env.total[i])+' -- Residual:' +str(env.residual[i])+ "SR: " +str(strehl)+'\n')
            
            
            if (i+1) % 500 == 0:
                sr = env.calculate_strehl_AVG()
                SRs.append(sr)
                rewards.append(accu_reward)
                accu_reward = 0
            
        print("SRs: ",SRs)
        print("Rewards",rewards)
        print("Saving Data")

        save_plots(savedir,SRs,rewards,env.LE_PSF) #savedir,evals,reward_sums,env.LE_PS
        print("Data Saved")