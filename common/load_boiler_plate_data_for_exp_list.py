##################################################################################
#   For a given analysis, we often need a bunch of information in regards to 
#   the experiments, the cell type label graph, the gene expression data. This 
#   function computes all of this 'boiler plate' data.
##################################################################################

import sys
import os
from os.path import join
import json
import collections
from collections import defaultdict

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import project_data_on_ontology as pdoo
import the_ontology
import featurizers

def load_everything_not_data(env_dir, exp_list_name):
    og = the_ontology.the_ontology()
    labels_f =  join(
        env_dir,
        "data",
        "experiment_lists",
        exp_list_name,
        "labelling.json"
    )
    exp_list_f = join(
        env_dir,
        "data",
        "experiment_lists",
        exp_list_name,
        "experiment_list.json"
    )
    data_set_metadata_f = join(
        env_dir,
        "data",
        "data_set_metadata.json"
    )

    with open(exp_list_f, 'r') as f:
        the_exps = json.load(f)['experiments']

    with open(data_set_metadata_f, 'r') as f:
        exp_to_info = json.load(f)

    # load the label graph
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
        jsonable_graph = labels_data['label_graph']
        jsonable_exp_to_labels = labels_data['labelling']
    label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    exp_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_exp_to_labels)
    exp_to_labels = {k:set(v) for k,v in exp_to_labels.iteritems()}

    # map each experiment to its ontology terms
    exp_to_terms = defaultdict(lambda: set())
    for exp, labels in exp_to_labels.iteritems():
        for label in labels:
            exp_to_terms[exp].update(label)

    # map each experiment to its study 
    exp_to_study = {
        exp: exp_to_info[exp]['study_accession']
        for exp in the_exps
    }
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.iteritems():
        study_to_exps[study].add(exp)
    study_to_exps = dict(study_to_exps)

    exp_to_tags = {
        exp: exp_to_info[exp]['tags']
        for exp in the_exps
    }

    # map each experiment to its most-specific labels
    exp_to_ms_labels = {
        exp: label_graph.most_specific_nodes(labels)
        for exp, labels in exp_to_labels.iteritems()
    }

    # map each label to its name
    label_to_name = {
        x: pdoo.convert_node_to_name(x, og)
        for x in label_graph.get_all_nodes()
    }

    return (
        og,
        label_graph,
        label_to_name,
        the_exps,
        exp_to_labels,
        exp_to_terms,
        exp_to_tags,
        exp_to_study,
        study_to_exps,
        exp_to_ms_labels
    )



def load_everything(env_dir, exp_list_name, data_features):
    r = load_everything_not_data(env_dir, exp_list_name)
    og = r[0]
    label_graph = r[1]
    label_to_name = r[2]
    the_exps = r[3]
    exp_to_labels = r[4]
    exp_to_terms = r[5]
    exp_to_tags = r[6]
    exp_to_study = r[7]
    study_to_exps = r[8]
    exp_to_ms_labels = r[9]

    # load data
    the_exps, data_matrix, gene_names = featurizers.featurize(
        data_features,
        the_exps,
        assert_all_data_retrieved=True
    )
    exp_to_index = {
        exp: exp_i
        for exp_i, exp in enumerate(the_exps)
    }
    the_exps_set = set(the_exps)
    print "%d total experiments in my dataset" % len(the_exps)

    return (
        og,
        label_graph,
        label_to_name,
        the_exps,
        exp_to_index,
        exp_to_labels,
        exp_to_terms,
        exp_to_tags,
        exp_to_study,
        study_to_exps,
        exp_to_ms_labels,
        data_matrix,
        gene_names
    )
        

if __name__ == "__main__":
    main()
