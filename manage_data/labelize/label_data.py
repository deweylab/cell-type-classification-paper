###############################################################################
#   Given a data set and set of partitions, create a label graph
#   and label the data according to that labeling graph and each experiment's
#   mapped terms in the MetaSRA.
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

#    exp_to_ms_terms = labelizers.base_term_labelize(
#        'most_specific_cell_types',
#        the_experiments,
#        exp_to_info
#    )
    og = the_og.the_ontology()

#    all_ms_terms = set()
#    for ms_terms in exp_to_ms_terms.values():
#        all_ms_terms.update(ms_terms) 
#    label_graph, exp_to_labels = pdoo.collapsed_ontology_graph(
#        og, 
#        all_ms_terms,
#        exp_to_terms
#    )

    onto_graph = ogst.ontology_subgraph_spanning_terms(
        all_terms,
        og
    )
    label_graph_source_to_targets = {
        frozenset([source]): set([
            frozenset([target])
            for target in targets
        ])
        for source, targets in onto_graph.source_to_targets.iteritems()
    }
    label_graph = graph.DirectedAcyclicGraph(label_graph_source_to_targets)
    label_graph = graph.transitive_reduction_on_dag(label_graph)   

#    onto_source_to_targets = {}
#    for term in all_terms:
#        onto_source_to_targets[frozenset([term])] = set([
#            frozenset([child])
#            for child in og.id_to_term[term].relationships['inv_is_a']
#        ])        
#    label_graph = graph.DirectedAcyclicGraph(onto_source_to_targets)
#    label_graph = graph.transitive_reduction_on_dag(label_graph)

    # Label all of the data
    exp_to_labels = label_graph_labelize(
        the_experiments, 
        exp_to_info, 
        label_graph,
        base_term_labelizer
    )

    # Write labelled-data and label-graph to output file
    jsonable_label_graph = pdoo.graph_to_jsonable_dict(label_graph)
    jsonable_all_exps_to_labels = pdoo.labelling_to_jsonable_dict(exp_to_labels)

    with open(join(out_dir, "labelling.json"), 'w') as  f:
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
   
 
def label_graph_labelize(
        exp_accs, 
        exp_to_info, 
        label_graph, 
        base_term_labelizer
    ):
    term_to_label = {}
    for label in label_graph.get_all_nodes():
        for term in label:
            term_to_label[term] = label

    exp_to_labels = defaultdict(lambda: set())
    exp_to_terms = labelizers.base_term_labelize(
        base_term_labelizer, 
        exp_accs, 
        exp_to_info
    ) 
    for exp, terms in exp_to_terms.iteritems():
        mapped_labels = set([
            term_to_label[term]
            for term in terms
            if term in term_to_label
        ])
        exp_to_labels[exp] = mapped_labels
    return exp_to_labels


def run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)
    

if __name__ == "__main__":
    main()
