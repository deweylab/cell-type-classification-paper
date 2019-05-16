###############################################################################
#   Given a data set and set of partitions, create a label graph
#   and label the data according to that labeling graph and each experiment's
#   mapped terms in the MetaSRA. The label DAG formed in this script is the
#   'collapsed' DAG. That is, it is not the original ontology, but rather, 
#   groups terms in the ontology together when those terms share the same data.
#   These grouped terms are the labels. An edge is formed from label A to B if
#   the data pertaining to label B is a subset of the data pertaining to A.
###############################################################################

import os
from os.path import join
import sys

from optparse import OptionParser
import json
import subprocess
from collections import defaultdict

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping")

import load_experiment_lists as lel
import project_data_on_ontology as pdoo
import labelizers
import the_ontology as the_og
import graph_lib
from graph_lib import graph
import ontology_graph_spanning_terms as ogst

ENVIRONMENTS_ROOT = "/tier2/deweylab/mnbernstein/phenotyping_environments"


def main():
    usage = "usage: %prog <config file>"
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

#    config_f = args[0]
#    with open(config_f, 'r') as f:
#        config = json.load(f)

#    env_name = config['env_name']
#    base_term_labelizer = config['base_terms']
#    experiment_list_name = config['experiment_list_name']
    env_root = args[0]
    experiment_list_name = args[1]    
    base_term_labelizer = args[2]

#    env_root = join(ENVIRONMENTS_ROOT, env_name)
    data_info_f = join(env_root, "data", "data_set_metadata.json")

    with open(data_info_f, 'r') as f:
        exp_to_info = json.load(f)

    the_experiments = lel.union_of_experiment_lists(
        env_root, 
        [experiment_list_name]
    )

    # Create output directory
    out_dir = join(env_root, "data", "experiment_lists", experiment_list_name)

    # Collapse ontology
    exp_to_terms = labelizers.base_term_labelize(
        base_term_labelizer, 
        the_experiments, 
        exp_to_info
    )

    all_terms = set()
    for terms in exp_to_terms.values():
        all_terms.update(terms)

    og = the_og.the_ontology()
    label_graph, exp_to_labels = pdoo.collapsed_ontology_graph(exp_to_terms)

    # Write labelled-data and label-graph to output file
    jsonable_label_graph = pdoo.graph_to_jsonable_dict(label_graph)
    jsonable_all_exps_to_labels = pdoo.labelling_to_jsonable_dict(exp_to_labels)

    with open(join(out_dir, "labelling_collapsed_dag.json"), 'w') as  f:
        f.write(json.dumps(
            {
                'labelling_config': {
                    'base_terms': base_term_labelizer,
                    'experiment_list_name': experiment_list_name
                },
                'label_graph': jsonable_label_graph,
                'labelling': jsonable_all_exps_to_labels
            },
            indent=4,
            separators=(',', ': ')
        ))
   
def run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)
    

if __name__ == "__main__":
    main()
