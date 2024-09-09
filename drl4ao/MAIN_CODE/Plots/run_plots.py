# -*- coding: utf-8 -*-
"""
Main code to call the plot functions
@author: Raissa Camelo (LAM) git: @srtacamelo
"""

from plots import *
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__=='__main__':
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir = f'C:/Users/rcamelo/Projects/LAM/results/{timestamp}'+'_AO4ELT7_'+'v1.2'+'_20s' 

    base_url = "C:/Users/rcamelo/Projects/LAM/logs/v1.2/"
    base_url_integrator = base_url+"integrator/"    
    base_url_drl = base_url+"po4ao/" 

    rename_folders(base_url_integrator)
    rename_folders(base_url_drl)

    
    results_tag = ["atmos_chaos_r0","atmos_dec_r0","atmos_inc_r0","atmos_chaos_ws","atmos_dec_ws","atmos_inc_ws"] 
    results_tag = ["atmos_dec_r0","atmos_inc_ws"] 
    results_tag = ["WS10_r013","WS20_r013","WS10_r08.6","WS20_r08.6"] 

    integrator_list, drl_list = fetch_folder_names(results_tag,experiment_tag="20s",gain_profiles=[0.3,0.9]) 

    print(integrator_list,drl_list)
    plot_all_mbrl_vs_integrator(savedir,base_url,integrator_list,drl_list,results_tag,gains = [0.3,0.9],save=True) #0.5,0.9,1.2

    #plot_quick_sr(savedir,base_url_drl,drl_list,results_tag,save=False)
    #plot_quick_sr(savedir,base_url_integrator,flatten_list(integrator_list),results_tag,save=True)
    #plot_all_gains_best_sr(savedir,base_url_integrator,integrator_list,results_tag,titles =["","","",""],gains = [0.1, 0.3,0.5, 0.7,0.9,0.95,1,1.1,1.2,1.5,1.7,2.0,2.5],labels=[["r0 = 0.2270m","r0 = 0.1622m","r0 = 0.1128m"],["WS = 10m/s","WS = 14m/s","WS = 20m/s"]]) 
    #plot_all_gains_average_sr(savedir,base_url_integrator,integrator_list,results_tag,titles=[],gains=[0.3,0.9]) # U plot

    # #plot_all(integrator_list,drl_list,results_tag)



