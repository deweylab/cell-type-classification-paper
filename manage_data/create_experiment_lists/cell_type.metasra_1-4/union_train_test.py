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
    parser.add_option("-o", "--out_file", help="Test set output experiment list file")
    (options, args) = parser.parse_args()

    train_exps_list_f = args[0]
    test_exps_list_f = args[1]
    out_f = options.out_file

    og = the_ontology.the_ontology()

    with open(train_exps_list_f, 'r') as f:
        train_include_experiments_data = json.load(f)
    with open(test_exps_list_f, 'r') as f:
        test_include_experiments_data = json.load(f)

    include_experiments = set(train_include_experiments_data['experiments'])
    include_experiments.update(
        set(test_include_experiments_data['experiments'])
    )
    train_parent_exp_list_name = train_include_experiments_data['list_name']
    test_parent_exp_list_name = test_include_experiments_data['list_name']

    assert train_parent_exp_list_name == "training_set_experiments"
    assert test_parent_exp_list_name == "test_set_experiments"

    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "untampered_bulk_primary_cells_with_data",
                "description": "These union of experiments in the experiment list '%s' and '%s'" % (
                    train_parent_exp_list_name,
                    test_parent_exp_list_name
                ),
                "experiments": list(include_experiments)
            },
            indent=4
        ))


if __name__ == "__main__":
    main()
