import argparse
import time
import sys,os
import subprocess
import numpy as np
from write_script import create_bash_script


## I should implement parameter file instead of arguments

#####################################################################################
parser = argparse.ArgumentParser(description="Wrapper")
parser.add_argument("-s", "--simulation", required=True, type=str, 
                    help="Simulation Type")
parser.add_argument("-b", "--observable", required=True,
                    choices=["smf", "smz", "smgf"],
                    nargs="+", help="Choose observables")
parser.add_argument("-n", "--name", required=True, type=str, 
                    help="Name of output folder")
parser.add_argument("-N", "--num_sim", default=50, type=int, 
                    help="Number of simulations per iteration")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose output")
args = parser.parse_args()


args_sam  = "-s " + args.simulation + ' '
args_sam += '-n ' + args.name + ' '

args_ili  = "-b "
for obs in args.observable:
    args_ili += obs + ' '
args_ili += '-n ' + args.name + ' '
args_ili += '-N ' + str(args.num_sim) + ' '
#####################################################################################
fpath      = '/mnt/home/yjo10/ceph/ILI/SAM/result/' + args.name
sam_path   = '/mnt/home/yjo10/ceph/ILI/SAM/one_click_sam'

try:
    os.system("mkdir {}".format(fpath))
    os.system("mkdir {}/sam".format(fpath))
    os.system("mkdir {}/params".format(fpath))
except:
    pass

try:
    f = open("{}/n_round.txt".format(fpath),"r")
    n_round = int(f.read())
except:
    n_round = 0
    if not os.path.isfile('{}/params/params_{}.npy'.format(fpath,n_round)):
        theta   = np.random.uniform(
            low =(0.25, 0.425, 1., 27.5, 0.6, 5e-4, 0.025),
            high=(4., 6.8, 5., 440., 2.4, 0.008, 0.4), size=(args.num_sim,7))
        np.save("{}/params/params_{}.npy".format(fpath,n_round),theta)
        with open("{}/n_round.txt".format(fpath), 'w') as f:
            f.write(str(int(n_round)))
    print("round 0 files for params are written!")


start_ili = False
start_sam = False
if os.path.isfile('{}/params/params_{}.npy'.format(fpath,n_round)):
    start_sam = True

if start_sam and os.path.isfile('{}/sam/{}_{}.npy'.format(fpath,args.observable[0],n_round)):
    start_ili = True
    start_sam = False

if not start_ili and not start_sam:
    print("There are no params files and sam files!")
    print("You might want to restart the program with lower n_round.")
    raise

time_delay_SAM    = 60 # 5 seconds
time_delay_ILI    = 60 #5*60 # 5 seconds
time_delay_queue  = 15 #5*60 # 5 seconds
time_delay_buffer = 20
time_delay_here   = 1
status            = None
job_running       = False
#####################################################################################
def check_obs(obs):
    if obs.shape[0] == 8:
        pass
#####################################################################################






print("Wrapper starts!")
while (True):

    # check if proposal from ILI is generated correctly
    if start_sam and os.path.isfile('{}/params/params_{}.npy'.format(fpath,n_round)):

        ## write a job script
        create_bash_script('template_sam.txt', f'{sam_path}/run.sh',
                           [str(args.num_sim), args_sam])

        ## Job submission for SAM
        queue       = subprocess.run(["./run_SAM.sh"], capture_output=True)
        stdout      = str(queue.stdout).split()
        JOBID       = int(stdout[3][:-3])
        print ("--SAM job submitted (JOBID = {})".format(JOBID))


        ## Job status for SAM
        while (True):
            queue_state = subprocess.run(["squeue","--user=yjo10",
                                          "--job={}".format(JOBID)],
                                         capture_output=True)
            try:
                status = queue_state.stdout.split()[12].decode('UTF-8')
                if status == "R" and job_running is False:
                    time_delay_here = time_delay_SAM
                    job_running     = True
                    print("----Job is successfully running.")
            except IndexError:  ## when job is already finished
                if os.path.isfile('{}/sam/smf_{}.npy'.format(fpath,n_round)):
                    time_delay_here = time_delay_queue
                    job_running     = False
                    print("----Job is successfully finished.")
                    start_ili = True
                    start_sam = False
                    break
                else:
                    print("Something might be wrong!")
                    raise
            time.sleep(time_delay_here)

    time.sleep(time_delay_buffer)



    # check if output from SAM is generated correctly
    if start_ili and \
       os.path.isfile('{}/sam/{}_{}.npy'.format(fpath, args.observable[0], n_round)):
        print("Round {}".format(n_round))
        print("--Data loaded")
        with open("{}/n_round.txt".format(fpath), "w") as f:
                f.write("{}".format(n_round))


        ## generate job script for ILI
        create_bash_script('template_ili.txt', 'run.sh', args_ili)

        ## Job submission for ILI
        chdir       = subprocess.run(["cd","/mnt/home/yjo10/ceph/ILI/SAM"], capture_output=True)
        queue       = subprocess.run(["sbatch","run.sh"], capture_output=True)
        stdout      = str(queue.stdout).split()
        JOBID       = int(stdout[3][:-3])
        print ("--ILI job submitted (JOBID = {})".format(JOBID))

        ## Job status for ILI
        while (True):
            queue_state = subprocess.run(["squeue","--user=yjo10",
                                          "--job={}".format(JOBID)],
                                         capture_output=True)
            try:
                status = queue_state.stdout.split()[12].decode('UTF-8')
                if status == "R" and job_running is False:
                    time_delay_here = time_delay_ILI
                    job_running     = True
                    print("----Job is successfully running.")
            except IndexError:  ## when job is already finished
                if os.path.isfile('{}/params/params_{}.npy'.format(fpath,n_round+1)):
                    time_delay_here = time_delay_queue
                    n_round         = n_round + 1
                    job_running     = False
                    print("----Job is successfully finished.")
                    start_ili = False
                    start_sam = True
                    break
                else:
                    print("Something might be wrong!")
                    raise
            time.sleep(time_delay_here)

        time.sleep(time_delay_buffer)


    if not start_ili and not start_sam:
        print("Both ILI and SAM do not run.")
        print("Program terminates.")
        raise


