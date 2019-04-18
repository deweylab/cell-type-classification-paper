####################################################################################
#   This script creates a training set and testing set for training and evaluating
#   machine learning algorithms. Both the train set and test set are created by
#   cross-referencing all of the 'untampered' primary cell experiments against a
#   precomputed partition of the studies into training studies and test studies.
#
#   By 'untampered' experiments, I mean all experiments that have been treated in 
#   such a way that a purposeful change in gene expression was induced.  This does 
#   not include experiments that have been transfected with a control vector.
####################################################################################

from optparse import OptionParser
import sys
import json
from collections import defaultdict 

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/graph_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import labelizers
import the_ontology
import graph
import kallisto_quantified_data_manager_hdf5 as kqdm

def main():
    usage = "usage: %prog <experiment metadata file> <untampered exp list file for experiments that have data> <train-test set partition file>" 
    parser = OptionParser(usage=usage)
    parser.add_option("-r", "--train_out_file", help="Training set output experiment list file")
    parser.add_option("-e", "--test_out_file", help="Test set output experiment list file")
    (options, args) = parser.parse_args()

    exp_info_f = args[0]
    untampered_exps_w_data_list_f = args[1]
    train_test_partition_f = args[2] 
    train_out_f = options.train_out_file
    test_out_f = options.test_out_file

    og = the_ontology.the_ontology()

    with open(untampered_exps_w_data_list_f, 'r') as f:
        include_experiments_data = json.load(f)
    with open(exp_info_f, 'r') as f:
        exp_to_info = json.load(f)
    with open(train_test_partition_f, 'r') as f:
        partition_data = json.load(f)

    train_studies = partition_data['train_set_studies']
    test_studies = partition_data['test_set_studies']

    include_experiments = set(include_experiments_data['experiments'])
    parent_exp_list_name = include_experiments_data['list_name']
    assert parent_exp_list_name == "untampered_bulk_primary_cells_with_data"

    exp_to_study = {
        exp: exp_to_info[exp]['study_accession']
        for exp in exp_to_info
    }
    study_to_exps = defaultdict(lambda: set())
    for exp in include_experiments:
        study = exp_to_study[exp]
        study_to_exps[study].add(exp) 

    train_exps = set()
    for study in train_studies:
         train_exps.update(study_to_exps[study])
    test_exps = set()
    for study in test_studies:
        test_exps.update(study_to_exps[study])

    with open(train_out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "training_set_experiments",
                "description": "These are a subset of the experiments in the experiment list '%s' cross referenced with the training set partition" % (
                    parent_exp_list_name
                ),
                "experiments": list(train_exps)
            },
            indent=4
        ))    
    with open(test_out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "test_set_experiments",
                "description": "These are a subset of the experiments in the experiment list '%s' cross referenced with the test set partition." % (
                    parent_exp_list_name
                ),
                "experiments": list(test_exps)
            },
            indent=4
        ))


if __name__ == "__main__":
    main()
