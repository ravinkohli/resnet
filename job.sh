#!/bin/bash
#SBATCH -p meta_gpu-x # partition (queue)
#SBATCH --mem 6000 # memory pool for all cores (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -o log/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J resnet9-self-david+128-clip # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"; 

# Job to perform
mkdir -p ./example/job- &&  python main.py 
exit $?

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";