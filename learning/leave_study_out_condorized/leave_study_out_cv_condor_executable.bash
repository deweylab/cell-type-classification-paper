#!/bin/bash
tar -zxf leave_study_out_cv_condor_tarball.tar.gz

export PYTHONPATH=/ua/mnbernstein/software/pip_installations/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=./leave_study_out_cv_condor_tarball:$PYTHONPATH
export PYTHONPATH=./leave_study_out_cv_condor_tarball/common:$PYTHONPATH
export PYTHONPATH=./leave_study_out_cv_condor_tarball/machine_learning:$PYTHONPATH
export PYTHONPATH=./leave_study_out_cv_condor_tarball/quadprog/lib/python2.7/site-packages:$PYTHONPATH

config=$1
env_dir=$2
exp_list_name=$3
studies_list_f=$4

mkdir ./artifacts
mv ./leave_study_out_cv_condor_tarball/leave_study_out_cv_condor_job.py .
~/software/python_environment/python_build/bin/python leave_study_out_cv_condor_job.py $config $env_dir $exp_list_name -r ./artifacts -a ./leave_study_out_cv_condor_tarball/algo_config -s $studies_list_f
