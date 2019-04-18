#############################################################
#   This is the script that is kicked off by the Condor
#   executable
#############################################################
from optparse import OptionParser
import sys
import os
from os.path import join
import math
import time

import json
import collections
from collections import defaultdict, Counter
import numpy as np
import subprocess
import joblib
from joblib import Parallel, delayed
import cPickle

import labelizers
import featurizers
import project_data_on_ontology as pdoo
import the_ontology
import leave_study_out as lso
import load_boiler_plate_data_for_exp_list as lbpdfel

PER_JOB_OUT_F = "leave_study_out_cv_results_batch.json"

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    parser.add_option("-a", "--algo_config_dir", help="The directory where all the classifier configurations are stored")
    parser.add_option("-r", "--artifacts_dir", help="The directory in which to write temporary files")
    parser.add_option("-s", "--studies_list_file", help="The list of studies to run in this batch")
    (options, args) = parser.parse_args()

    config_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    algo_config_dir = options.algo_config_dir
    studies_list_f = options.studies_list_file
    artifacts_dir = options.artifacts_dir

    with open(config_f, 'r') as f:
        config = json.load(f)

    with open(studies_list_f, 'r') as f:
        batch_studies = json.load(f)

    features = config['features']
    collapse_ontology = config['collapse_ontology']
    if 'normalize_polyA_total' in config:
        normalize_polyA_total = config['normalize_polyA_total']
    else:
        normalize_polyA_total = False


    # use a helper function to load all of the data
    r = lbpdfel.load_everything(env_dir, exp_list_name, features)
    og = r[0]
    label_graph = r[1]
    label_to_name = r[2]
    the_exps = r[3]
    exp_to_index = r[4]
    exp_to_labels = r[5]
    exp_to_terms = r[6]
    exp_to_tags = r[7]
    exp_to_study = r[8]
    study_to_exps = r[9]
    exp_to_ms_labels = r[10]
    data_matrix = r[11]
    gene_names = r[12]
    print "%d total experiments in my dataset" % len(the_exps)

    # load classifier configuration
    algo_config_f = join(
        algo_config_dir,
        config['algo_config_file']
    )
    with open(algo_config_f, 'r') as f:
        algo_config = json.load(f)
    classif_name = algo_config['algorithm']
    classif_params = algo_config['params']

    # load dimension reduction algorithm
    dim_reduct_config_f = None
    if 'dim_reduct_config_file' in config:
        dim_reduct_config_f = config['dim_reduct_config_file']
    dim_reduct_name = None
    dim_reduct_params = None
    if dim_reduct_config_f:
        with open(dim_reduct_config_f, 'r') as f:
            dim_reduct_config = json.load(f)
            dim_reduct_name = dim_reduct_config['algorithm']
            dim_reduct_params = dim_reduct_config['params']

    # Run the fold
    exp_to_label_str_to_conf = {}
    for study_i, leave_out_batch in enumerate(batch_studies):
        leave_out_study = leave_out_batch['study']
        test_on_experiments = leave_out_batch['experiments']

        print "Running fold %d/%d in batch..." % (
            study_i,
            len(batch_studies)
        )
        leave_out_exp_to_label_str_to_conf, leave_out_exp_to_label_str_to_score = lso.leave_study_out(
            leave_out_study,
            test_on_experiments,
            data_matrix,
            the_exps,
            study_to_exps,
            exp_to_study,
            exp_to_terms,
            label_graph,
            exp_to_labels,
            og,
            classif_name,
            classif_params,
            collapse_ontology,
            gene_names,
            dim_reduct_name=dim_reduct_name,
            dim_reduct_params=dim_reduct_params,
            artifacts_parent_dir=artifacts_dir,
            exp_to_tags=exp_to_tags,
            normalize_polyA_total=normalize_polyA_total
        )
        exp_to_label_str_to_conf.update(
            leave_out_exp_to_label_str_to_conf
        )

    with open(PER_JOB_OUT_F, 'w') as f:
        f.write(
            json.dumps(
                {
                    "batch_studies": batch_studies,
                    "predictions": exp_to_label_str_to_conf
                },
                indent=4,
                separators=(',', ': ')
            )
        )
    

if __name__ == "__main__":
    main()
