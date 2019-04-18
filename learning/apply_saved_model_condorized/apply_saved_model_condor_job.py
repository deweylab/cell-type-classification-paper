#########################################################################
#   Train a hiearchical classifier on an experiment list and pickle the
#   model
#########################################################################

import sys
import os
from os.path import join, basename

from optparse import OptionParser
import json
import collections
from collections import defaultdict, Counter
import numpy as np
import cPickle

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/learning")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import featurizers
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from machine_learning import learners
import load_boiler_plate_data_for_exp_list as lbpdfel
import project_data_on_ontology as pdoo


def main():
    usage = "usage: %prog <config file> <environment dir> <experiment list name>"
    parser = OptionParser(usage)
    parser.add_option("-r", "--artifacts_parent_dir", help="The directory in which to write temporary files")
    (options, args) = parser.parse_args()

    model_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    exp_list_f = args[3]

    print "environment directory: %s" % env_dir
    print "experiment list: %s" % exp_list_name

    with open(exp_list_f, 'r') as f:
        apply_to_exps = json.load(f)

    # load the data
    r = lbpdfel.load_everything_not_data(
        env_dir, 
        exp_list_name
    )
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

    the_exps, data_matrix, gene_names = featurizers.featurize(
        'kallisto_gene_log_cpm',
        apply_to_exps,
        assert_all_data_retrieved=True
    )
    exp_to_index = {
        exp: exp_i
        for exp_i, exp in enumerate(the_exps)
    }

    print "loading model..."
    with open(model_f, 'r') as f:
        model = cPickle.load(f)
    print "done."

    assert tuple(gene_names) == tuple(model.classifier.feat_names)

    # Returns a list of dictionaries. Each dictionary corresponds
    # to a sample and maps a label/node to its confidence of being
    # assigned
    node_to_confidences, node_to_classif_scores = model.predict(data_matrix)

    exp_to_label_str_to_conf = {}
    for exp, node_to_confidence in zip(the_exps, node_to_confidences):
        exp_to_label_str_to_conf[exp] = {
            pdoo.convert_node_to_str(node): conf
            for node, conf in node_to_confidence.iteritems()
        }

    out_f = "apply_saved_model_results_batch.json"
    with open(out_f, 'w') as f:
        f.write(json.dumps(
            exp_to_label_str_to_conf,
            indent=4
        ))

if __name__ == "__main__":
    main()
