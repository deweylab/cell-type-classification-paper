#!/bin/bash

# Inputs:
#   1. The name of the training-configuration file
#   2. The directory storing the dataset 
#   3. The file storing the job's input information
#   4. The name of the file to write output to
config=$1
dataset_dir=$2
fold_f=$3
out_f=$4

# Unpack the input
tar -zxf condor_tarball.tar.gz

# Setup environment
#source /ua/mnbernstein/software/use_python3.sh
export PYTHONPATH=/ua/mnbernstein/software/python3_env/lib/python3.6/site-packages
export PYTHONPATH=/ua/mnbernstein/projects:/ua/mnbernstein/projects/tbcp/cello_dev:$PYTHONPATH
echo $PYTHONPATH

# Run the master script
python3 ./condor_tarball/condor_job.py ./condor_tarball/training_parameter_sets/$config $dataset_dir $fold_f -o $out_f
