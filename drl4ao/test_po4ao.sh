#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                        # memory per node
#SBATCH --time=00-02:59         # time (DD-HH:MM)
#SBATCH -e ../../logs/unmod_po4ao/err_po4ao.txt
#SBATCH --account=def-lplevass
#SBATCH --job-name=wf_unmod_po4ao
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=parker.levesque@gmail.com

cd MAIN_CODE

savedir='unmod_po4ao'
logpath="../../logs/"$savedir
mkdir -p $logpath 

source $HOME/projects/def-lplevass/parker09/drl4ao_env/bin/activate
python $HOME/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/mbrl_main_network.py > ../../logs/$savedir/po4ao_out.txt 2>&1
#python $HOME/projects/def-lplevass/parker09/RLAO/drl4ao/MAIN_CODE/integrator_oopao_razor.py > ../../logs/$savedir/int_out.txt 2>&1


