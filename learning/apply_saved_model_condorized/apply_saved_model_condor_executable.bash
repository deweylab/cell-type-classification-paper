#!/bin/bash
tar -zxf apply_saved_model_condor_tarball.tar.gz

export PYTHONPATH=/ua/mnbernstein/software/pip_installations/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=./apply_saved_model_condor_tarball:$PYTHONPATH
export PYTHONPATH=./apply_saved_model_condor_tarball/common:$PYTHONPATH
export PYTHONPATH=./apply_saved_model_condor_tarball/machine_learning:$PYTHONPATH
export PYTHONPATH=./apply_saved_model_condor_tarball/quadprog/lib/python2.7/site-packages:$PYTHONPATH

model_f=$1
env_dir=$2
exp_list_name=$3
exp_list_f=$4

mv ./apply_saved_model_condor_tarball/apply_saved_model_condor_job.py .
~/software/python_environment/python_build/bin/python apply_saved_model_condor_job.py $model_f $env_dir $exp_list_name $exp_list_f
