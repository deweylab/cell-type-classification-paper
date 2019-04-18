#!/bin/bash
tar -zxf gsea_coeffs_condor_tarball.tar.gz

export PYTHONPATH=/ua/mnbernstein/software/pip_installations/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=./gsea_coeffs_condor_tarball:$PYTHONPATH

mv ./gsea_coeffs_condor_tarball/gsea_coeffs_condor_job.py .

label_f=$1
gmt_f=gsea_coeffs_condor_tarball/Entrez_GO_sets.biological_function.gmt

mv ./leave_study_out_cv_condor_tarball/gsea_coeffs_condor_job.py .
~/software/python_environment/python_build/bin/python gsea_coeffs_condor_job.py $label_f $gmt_f
