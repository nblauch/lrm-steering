#!/bin/bash -l

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nblauch@fas.harvard.edu
#SBATCH --mem=256GB
#SBATCH --account=kempner_alvarez_lab
#SBATCH --cpus-per-task 36
#SBATCH -p kempner

cd ~/git/lrm-steering

echo "$COMMAND"
$COMMAND
