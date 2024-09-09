# -*- coding: utf-8 -*-
"""
Plotting functions for checking results and conference papers
@author: Raissa Camelo (LAM) git: @srtacamelo
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch
import os

def flatten_list(list):
   """Transforms a given list of lists in a single list
   """
   flat_list = [item for sublist in list for item in sublist]
   return flat_list
def plot_sr_simple(SR_list,frames, y,savedir,tag = "",warmup = 1,save=True):
  """Simply plots the SR avarage over each episode (500 frames generally) for a simulation.
  Used for quickly checking simulation results (integrator or RL).
  """
  frames = range(frames)
  plt.plot(frames, SR_list,label='SR ',color='blue')
  plt.axvline(x=warmup, color='gray', linestyle='--')
  plt.text((warmup+0.2),y,'warm up',rotation=0, color='gray')
  # Add title and axis names
  plt.title(tag)
  plt.xticks(np.arange(0, len(frames)+1, 5))
  plt.yticks(np.arange(0, 90, 10))

  plt.xlim([0,len(frames)+1])
  plt.ylim([0,80])

  plt.ylabel('Strehl ratio (%)')
  plt.xlabel('Episodes (500 frames each)')

  # Extra info
  plt.grid()
  plt.legend(loc=4)
  if save:
   os.makedirs(savedir, exist_ok=True)
   plt.savefig(os.path.join(savedir, "sr_plot_"+tag+ "_"+".jpg"))
  plt.show()
  
def plot_sr(SR_integrator, SR_drl,frames, y,savedir,tag,gain,title = "",warmup = 1):
  """
  Plots a SR comparision between the PO4AO and the integrator at a given Gain
  """
  os.makedirs(savedir, exist_ok=True)
  frames = range(frames)
  plt.plot(frames, SR_integrator,label='Integrator gain: '+str(gain),color='blue')
  plt.plot(frames, SR_drl, label='PO4AO',color='red')
  plt.axvline(x=warmup, color='gray', linestyle='--')
  plt.text((warmup+0.2),y,'warm up',rotation=0, color='gray')
  # Add title and axis names
  plt.title(tag)
  plt.xticks(np.arange(0, len(frames)+1, 5))
  #plt.yticks(np.arange(0, 1, 0))
  plt.ylabel('Strehl ratio (%)')
  plt.xlabel('Episodes (500 frames each)')

  # Extra info
  plt.grid()
  plt.legend(loc=4)
  plt.savefig(os.path.join(savedir, "sr_plot_"+tag+ "_"+str(gain)+".jpg"))
  plt.show()

def plot_reward(rewards_integrator,rewards_drl,frames, y,savedir,tag,gain,title,warmup = 1):
  """Plots a  reward comparision between the PO4AO and the integrator at a given Gain
  """
  os.makedirs(savedir, exist_ok=True)
  frames = range(frames)
  plt.plot(frames, rewards_integrator,label='Integrator gain: '+str(gain),color='blue')
  plt.plot(frames, rewards_drl, label='PO4AO',color='red')
  plt.axvline(x=warmup, color='gray', linestyle='--')
  plt.text((warmup+0.2),y,'warm up',rotation=0, color='gray')
  # Add title and axis names
  plt.title(title)
  plt.xticks(np.arange(0, len(frames)+1, 5))
  plt.ylabel('Reward')
  plt.xlabel('Episode (500 frames each)')
  

  # Extra info
  plt.grid()
  plt.legend(loc=4)
  plt.savefig(os.path.join(savedir, "reward_plot_"+tag+ "_"+str(gain)+".jpg"))
  plt.show()

def plot_reward2(rewards_integrator_1,rewards_integrator_2,rewards_integrator_3,rewards_integrator_4,rewards_integrator_5,rewards_drl,frames, y,savedir,tag,warmup = 2):
  """
  if not os.path.exists(savedir):
     os.mkdir(savedir)
  """
  os.makedirs(savedir, exist_ok=True)

  frames = range(1,frames+1)
  plt.plot(frames, rewards_integrator_1,label='Integrator gain: 0.1',color='orange')
  plt.plot(frames, rewards_integrator_2,label='Integrator gain: 0.3',color='yellow')
  plt.plot(frames, rewards_integrator_3,label='Integrator gain: 0.5',color='green')
  plt.plot(frames, rewards_integrator_4,label='Integrator gain: 0.7',color='magenta')
  plt.plot(frames, rewards_integrator_5,label='Integrator gain: 0.9',color='blue')
  plt.plot(frames, rewards_drl, label='PO4AO',color='red')
  plt.axvline(x=warmup, color='gray', linestyle='--')
  plt.text((warmup+0.2),y,'warm up',rotation=0, color='gray')
  # Add title and axis names
  #plt.title('My title')
  plt.xticks(np.arange(1, len(frames)+1, 1))
  #plt.yticks(np.arange(0, max(rewards_drl), 50))
  plt.ylabel('Reward (nm)')
  plt.xlabel('Episode (500 frames each)')

  plt.xlim([1,len(frames)+1])
  #plt.ylim([0,max(rewards_drl)])

  # Extra info
  plt.grid(False)
  plt.legend(loc=4)
  #plt.savefig(os.path.join(savedir, "reward_plot_"+tag+".jpg"))
  plt.show()


def plot_values(values,frames,iterations,y,savedir,tag,title,label,gain="",warmup = 500):
  """Generic function created to plot r0 and Wind Speed values saved through a simulation for debbuging purpuses.
   Can be used to plot any generic graphic of values over Frames for a simulation.
  """
  os.makedirs(savedir, exist_ok=True)
  frames = range(frames)
  print(len(values),"values")
  plt.plot(frames, values,label=title+" "+str(gain),color='blue',marker="+",linestyle='None')
  #plt.plot(frames, rewards_drl, label='PO4AO',color='red')
  plt.axvline(x=warmup, color='gray', linestyle='--')
  plt.text((warmup+0.2),y,'warm up',rotation=0, color='gray')
  # Add title and axis names
  plt.title(title+" "+ tag)
  plt.xticks(np.arange(0,len(frames)+1, 500),np.arange(0,iterations+1))
  plt.ylabel(label)
  plt.xlabel('Episode (500 frames each)')

# Extra info
  plt.grid()
  plt.legend(loc=4)
  #plt.savefig(os.path.join(savedir, label+"_plot_"+tag+".jpg"))
  plt.show()

def plot_sr_multiple_gains_vs_drl(SR_integrator,SR_drl,frames, y,savedir,tag,gains=[0.3,0.9],warmup = 2,colors =['yellow','blue','magenta','orange','green'],save=True):
  """Function that plots multiple integrator SR curves compared to an RL SR curve.
  """
  frames = range(1,frames+1)
  for i in range(len(SR_integrator)):
   plt.plot(frames, SR_integrator[i],label='Integrator gain: '+str(gains[i]),color=colors[i])

  plt.plot(frames, SR_drl, label='PO4AO',color='red')
  plt.axvline(x=warmup, color='gray', linestyle='--')
  plt.text((warmup+0.2),y,'warm up',rotation=0, color='gray')
  # Add title and axis names
  plt.title(tag)
  plt.xticks(np.arange(1, len(frames)+1, 1))
  plt.yticks(np.arange(0, 90, 10))

  plt.xlim([1,len(frames)+1])
  plt.ylim([0,80])

  plt.ylabel('Strehl ratio (%)')
  plt.xlabel('Episode (500 frames each)')

  # Extra info
  plt.grid(False)
  plt.legend(loc=4)
  if save:
   print("Saved")
   os.makedirs(savedir, exist_ok=True)
   plt.savefig(os.path.join(savedir, "sr_plot_"+tag+".jpg"))
  plt.show()

def plot_sr_AVG_vs_gain(SR_integrator_AVG,gains,savedir,tag,color):
  os.makedirs(savedir, exist_ok=True)
  y = list(np.arange(1,len(SR_integrator_AVG)+1))
  plt.plot(y, SR_integrator_AVG,'--o',color=color)
  #my_xticks = ['a', 'b', 'c', 'd']
  plt.xticks(y, gains)

  plt.ylabel('Strehl ratio AVG (%)')
  plt.xlabel('Integrator Gain')
  #plt.title(tag)
 
  plt.grid(False)
  plt.savefig(os.path.join(savedir, "sr_plot_"+tag+ "_all_gains"+".jpg"))
  plt.show()

def plot_sr_curves_vs_gain(SR_best,SR_middle,SR_bottom,gains,savedir,tag,color,labels):
  os.makedirs(savedir, exist_ok=True)
  y = list(np.arange(1,len(SR_best)+1))
  plt.plot(y, SR_best,'--o',label=labels[0],color=color[0])
  plt.plot(y, SR_middle,'--o',label=labels[1],color=color[1])
  plt.plot(y, SR_bottom,'--o',label=labels[2],color=color[2])
  #my_xticks = ['a', 'b', 'c', 'd']
  plt.xticks(y, gains)

  plt.ylabel('Strehl ratio (%)')
  plt.xlabel('Integrator Gain')
  #plt.title(tag)

  plt.legend(loc=1)
  plt.grid(False)
  plt.savefig(os.path.join(savedir, "sr_plot_"+tag+ "_all_gains"+".jpg"))
  plt.show()
  
def read_logs(file_name):
  """ Read the simulation results to plot the graphs"""
  import torch
  re_file = torch.load(file_name+"/rewards2plot.pt")
  sr_file =  torch.load(file_name+"/sr2plot.pt")
  r0_file = []
  ws_file = []

  if os.path.exists(file_name+"/r02plot.pt"):
   r0_file = torch.load(file_name+"/r02plot.pt")
  if os.path.exists(file_name+"/ws2plot.pt"):
   ws_file =  torch.load(file_name+"/ws2plot.pt")
  
  
  if type(re_file) == list:
     rewards = re_file
  else:
     rewards = re_file.tolist()
  if type(sr_file) == list:
     SR = sr_file
  else:
     SR = sr_file.tolist()
  if type(r0_file) == list:
     r0 = r0_file
  else:
     r0 = r0_file.tolist()
  if type(ws_file) == list:
     ws = ws_file
  else:
     ws= ws_file.tolist()

  iterations = len(SR)

  return SR, rewards, r0,ws,iterations

def make_gif(frame_folder,gif_name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save(gif_name, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
def save_PSFs(PSFs, savedir,filename,psf_folder,psf_file):
    """ Saves ALL PSF images (Long or short exposure).
        Also intended to create a gif with all saved PSF images to demonstrate the convergion of the algorithm
        (closing the loop).
    """
    #from array2gif import write_gif
    from PIL import Image
    os.mkdir(os.path.join(savedir, psf_folder))
    i = 0
    rgb_psfs = []
    #rgb_psfs = [(Image.fromarray( (img* 255).astype(np.uint8), mode="F") ) for img in PSFs]
    
    for psf in PSFs:
        plt.imsave(os.path.join(savedir, psf_folder,psf_file+str(i)+".jpg"),psf)
        i += 1
        #psf = plt.imshow(psf)
        #plt.close()
        rgb_psfs.append(psf)
def save_txt(savedir,list,file_name,fmt="%10.5f"):
   if not os.path.exists(savedir):
      os.mkdir(savedir)
   np.savetxt(os.path.join(savedir, file_name),list, delimiter=',',fmt=fmt)

def save_plots(savedir,sr,rewards,LE_PSF,r0_list=[],ws_list=[]):
    """ Save results to plot graphs later and images of long and short exposure PSF.
    """
    if not os.path.exists(savedir):
      os.mkdir(savedir)

    torch.save(rewards, os.path.join(savedir, "rewards2plot.pt"))
    torch.save(sr, os.path.join(savedir, "sr2plot.pt"))
    np.savetxt(os.path.join(savedir, "sr2check.txt"), sr, delimiter=',')
    plt.imsave(os.path.join(savedir, "LE_PSF.jpg"),LE_PSF)
    
    if len(r0_list) > 0:
       print(r0_list)
       torch.save(r0_list, os.path.join(savedir, "r02plot.pt"))
       np.savetxt(os.path.join(savedir, "r02check.txt"), r0_list, delimiter=',')
    if len(ws_list) > 0:
       torch.save(ws_list, os.path.join(savedir, "ws2plot.pt")) 
       np.savetxt(os.path.join(savedir, "ws2check.txt"), ws_list, delimiter=',')     

def multiply_sr(sr):
    sr = [x * 100 for x in sr]
    return sr
def plot_all(savedir,base_url,integrator_list,drl_list,results_tag, gains = [0.3,0.9]):
    """
    Plot comparing Integrator SR with DRL4AO
    """
    for i in range(len(drl_list)):
        drl_file = drl_list[i]
        integrator_file = integrator_list[i]
        tag = results_tag[i]
        for j in range(len(integrator_file)):
            gain = gains[j]
            integrator_gain = integrator_file[j]
            SR_drl, rewards_drl,r0,ws, iterations = read_logs(base_url+"po4ao/"+drl_file)
            SR_integrator, rewards_integrator,r0,ws, iterations = read_logs(base_url+"integrator/"+integrator_gain)
        
            # print(len(SR_drl), len(rewards_drl), iterations)
            # print(len(SR_integrator), len(rewards_integrator), iterations)
            plot_sr(multiply_sr(SR_integrator),multiply_sr(SR_drl),iterations,np.max(SR_drl),savedir,tag,gain)
            #plot_reward(rewards_integrator,rewards_drl,iterations,np.max(rewards_drl),savedir,tag)

def plot_all_mbrl_vs_integrator(savedir,base_url,integrator_list,drl_list,results_tag,gains = [1.2,1.5,1.7,2.0,2.5],save=True):

    for i in range(len(drl_list)):
        drl_file = drl_list[i]
        integrator_file = integrator_list[i]
        tag = results_tag[i]
        
        SR_drl, rewards_drl,r0_drl,ws_drl, iterations = read_logs(base_url+"po4ao/"+drl_file)
        SR_integrator, rewards_integrator = [], []
        r0_integrator, ws_integrator = [], []

        for gain_profile in integrator_file:
            sr, rewards,r0,ws, iterations = read_logs(base_url+"integrator/"+gain_profile)
            r0_integrator.append(r0), ws_integrator.append(ws)
            SR_integrator.append(multiply_sr(sr)), rewards_integrator.append(rewards)
   
        print(tag)

        #plot_all_r0_or_ws(r0_integrator,r0_drl,iterations,tag,gains,"r0")
        #plot_all_r0_or_ws(ws_integrator,ws_drl,iterations,tag,gains,"ws")
        
        #plot_all_individual(SR_integrator,SR_drl,rewards_integrator,rewards_drl,iterations,tag,gains)
        #plot_sr3(multiply_sr(SR_integrator[0]),multiply_sr(SR_integrator[1]),multiply_sr(SR_integrator[2]),multiply_sr(SR_integrator[3]),multiply_sr(SR_integrator[4]),multiply_sr(SR_drl),iterations,((np.max(SR_drl))*100)-4,savedir,tag)
        plot_sr_multiple_gains_vs_drl(SR_integrator,multiply_sr(SR_drl),iterations,((np.max(SR_drl))*100)-4,savedir,tag,gains=gains,save=save)
        #plot_reward2(rewards_integrator[0],rewards_integrator[1],rewards_integrator[2],rewards_integrator[3],rewards_integrator[4],rewards_drl,iterations,np.max(rewards_drl),savedir,tag)

def plot_all_gains_average_sr(savedir,base_url,integrator_list,results_tag,titles,gains): 
    corlors = ["red","yellow","blue","green","brown","purple"]  
    for i in range(len(integrator_list)):
        integrator_file = integrator_list[i]
        #title = titles[i]
        tag = results_tag[i]
        color = corlors[i]
        SR_integrator_AVG, rewards_integrator = [], []
        for gain_profile in integrator_file:
            sr, rewards,r0,ws, iterations = read_logs(base_url+gain_profile)
            SR_integrator_AVG.append(np.average(sr)), rewards_integrator.append(rewards)
        #print(SR_integrator_AVG)
        #print(tag)
        plot_sr_AVG_vs_gain(multiply_sr(SR_integrator_AVG),gains,savedir,tag,color)
        # print(len(SR_drl), len(rewards_drl), iterations)
        # print(len(SR_integrator), len(rewards_integrator), iterations)
def plot_all_gains_best_sr(savedir,base_url,integrator_list,results_tag,titles,gains,labels): 
    corlors = [["red","green","blue"],["red","green","blue"]]
    for i in range(len(integrator_list)):
        integrator_file = integrator_list[i]
        #title = titles[i]
        tag = results_tag[i]
        color = corlors[i]
        label = labels[i]
        SR_integrator_best, SR_middle,SR_bottom, rewards_integrator = [], [],[],[]
        for gain_profile in integrator_file:
            sr, rewards,r0,ws, iterations = read_logs(base_url+gain_profile)
            SR_integrator_best.append(np.max(sr)), rewards_integrator.append(rewards)
            SR_middle.append(sr[2]),SR_bottom.append(sr[5])
        #print(SR_integrator_AVG)
        #print(tag)
        plot_sr_curves_vs_gain(multiply_sr(SR_integrator_best),multiply_sr(SR_middle),multiply_sr(SR_bottom),gains,savedir,tag,color,labels=label)
        #plot_sr_AVG_vs_gain(multiply_sr(SR_integrator_best),gains,savedir,tag,color)
        #plot_sr_AVG_vs_gain(multiply_sr(SR_first),gains,savedir,'first'+"_"+tag,color)
        #plot_sr_AVG_vs_gain(multiply_sr(SR_last),gains,savedir,'middle'+"_"+tag,color)
        # print(len(SR_drl), len(rewards_drl), iterations)
        # print(len(SR_integrator), len(rewards_integrator), iterations)
def plot_all_individual(savedir,SR_integrator,SR_drl,rewards_integrator,rewards_drl,episodes,tag,gains):
    for i in range(len(SR_integrator)):
        gain = gains[i]
        if len(SR_integrator[i]) == len(SR_drl):
            
            plot_sr(multiply_sr(SR_integrator[i]), multiply_sr(SR_drl),episodes, ((np.max(SR_integrator[i]))*100)-4,savedir,tag,gain,tag,warmup = 1)
            #plot_reward(rewards_integrator[i],rewards_drl,episodes, ((np.max(rewards_integrator[i]))*100),savedir,tag,gain,tag,warmup = 1)
        else:
            print("Problem with: "+tag+" drl len: "+str(len(SR_drl))+" integrator len: "+str(len(SR_integrator[i])))
def plot_all_r0_or_ws(values_integrator,values_drl,iterations,tag,gains,value_type):
    """Plot r0 or WS values for checking up the results.
    """
    for i in range(len(values_integrator)):
        if len(values_integrator[i]) > 0:
            gain = gains[i]
            if not (np.array_equal(values_integrator[i],values_drl)):
                print("Arrays are different. Problem with: "+tag+" drl len: "+str(len(values_drl))+" integrator len: "+str(len(values_integrator[i])))
                #print(values_integrator[i])
                #print(values_drl)

                if len(values_integrator[i]) != len(values_drl): 
                    print("Problem with: "+tag+" drl len: "+str(len(values_drl))+" integrator len: "+str(len(values_integrator[i])))
            #plot_values(values_integrator[i][4000:4500],500,1, (np.max(values_integrator[i])-4),savedir,tag,"Integrator",value_type,gain=gain)
            plot_values(values_integrator[i],len(values_integrator[i]),iterations, (np.max(values_integrator[i])-4),savedir,tag,"Integrator",value_type,gain=gain)
    if len(values_drl) > 0:
        pass
        #plot_values(values_drl,len(values_drl),iterations, ((np.max(values_drl)))-4,savedir,tag,"PO4AO",value_type,gain="")
def plot_quick_sr(savedir,base_url,experiments,tags,save=True):
    """Quickly plot the SR for all experiments on a list of experiments.

    """
    for i in range(len(experiments)):
        SR_list, rewards_drl,r0,ws,episodes = read_logs(base_url+experiments[i]) #+"/rewards2plot.pt",base_url+experiments[i]+"/sr2plot.pt"
        plot_sr_simple(multiply_sr(SR_list),episodes,((np.max(SR_list))*100)-4,savedir,tag=experiments[i],save=True)
        #plot_values(r0,len(r0),episodes,((np.max(r0))*100)-4,savedir,tags[i],"r0",experiments[i])


def fetch_folders(base_url_integrator,results_tag):
    integrator_list = []
    for tag in results_tag:
        folders = os.listdir(base_url_integrator)# +tag
        print(folders)
        integrator_list.append(folders)
    return integrator_list

def fetch_folder_names(base_experiment_tags=[],experiment_tag ="",gain_profiles= [0.9]):
    """This function fetches the names of the folders containing the integrator and the MBRL experiments
    that we want to get the data from for plotting. It's done automatically by the simulation code "tag".
    *WARNING: If you just want to plot integrator only or MBRL only results you still need to have both folders
    to call the rename_folders function (even if one of them is empty). Running this function on a empty folder will
    still result in the experiment tags being generated (even if the respective experiment folders do not exist)*
    """
    integrator_list = []
    mbrl_list = []
    for tag in base_experiment_tags:
        mbrl_experiment = "po4ao_"+tag +"_"+experiment_tag
        mbrl_list.append(mbrl_experiment)

        integrator_sublist = []
        for gain in gain_profiles:
            integrator_experiment = "i_"+tag+"_"+experiment_tag+"_"+str(gain)
            integrator_sublist.append(integrator_experiment)

        integrator_list.append(integrator_sublist)

    return integrator_list, mbrl_list
def rename_folders(main_path):
    """Renames the simulation folders resulted from the cluster. Removes the date YYYY/MM/DD and leaves only the
    standard simualtion labels, enabling the automatic plotting call with the fetch_folder_names function.
    *WARNING: If you just want to plot integrator only or MBRL only results you still need to have both folders
    to call the rename_folders function (even if one of them is empty).*
    """
    for filename in os.listdir(main_path):
        if filename.startswith("2"):
            print(filename[16: len(filename)]) 
            os.rename(main_path+filename,main_path+filename[16: len(filename)]) 




def list_from_dir(exp_root_dir, name_of_file):
    
    exp_dirs = os.listdir(exp_root_dir)

    values_list = []

    for subdir in exp_dirs:
        tmp = torch.load(exp_root_dir+'/'+subdir+'/'+name_of_file)

        if isinstance(tmp, torch.Tensor):
            tmp = tmp.numpy()

        values_list.append(tmp)

    return values_list
        



def plot_reward3(all_rewards,frames, gains, legend=False, style='seaborn-v0_8'):


    frames = range(1,frames+1)
    # colors = plt.cm.viridis(np.linspace(0, 1, len(all_rewards)))
    plt.style.use(style)

    for i in range(len(all_rewards)):
        plt.plot(frames, all_rewards[i],label=f'Integrator gain: {gains[i]}')#,color=colors[i])

    # Add title and axis names
    #plt.title('My title')
    plt.xticks(np.arange(1, len(frames)+1, 1))
    #plt.yticks(np.arange(0, max(rewards_drl), 50))
    plt.ylabel('Reward (nm)')
    plt.xlabel('Episode (500 frames each)')

    plt.xlim([1,len(frames)+1])
    #plt.ylim([0,max(rewards_drl)])

    # Extra info
    plt.grid(False)
    
    if legend:
        plt.legend(loc=4)
    #plt.savefig(os.path.join(savedir, "reward_plot_"+tag+".jpg"))
    plt.show()
