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

import experiment
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from machine_learning import learners
import load_boiler_plate_data_for_exp_list as lbpdfel
import project_data_on_ontology as pdoo


def main():
    usage = "usage: %prog <config file> <environment dir> <experiment list name>"
    parser = OptionParser(usage)
    parser.add_option("-o", "--out_dir", help="Directory in which to write the pickled model")
    parser.add_option("-r", "--artifacts_parent_dir", help="The directory in which to write temporary files")
    (options, args) = parser.parse_args()

    model_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    out_dir = options.out_dir

    print "environment directory: %s" % env_dir
    print "experiment list: %s" % exp_list_name

    # load the data
    r = lbpdfel.load_everything(
        env_dir, 
        exp_list_name,
        'kallisto_gene_log_cpm'         
    )
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

    print "loading model..."
    with open(model_f, 'r') as f:
        model = cPickle.load(f)
    print "done."

    print "Gene names from database: %s" % str(tuple(gene_names))
    print "Gene names from model: %s" % str(tuple(model.classifier.feat_names))
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

    out_f = join(out_dir, "prediction_results.json")
    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "predictions": exp_to_label_str_to_conf
            },
            indent=4
        ))


   

if __name__ == "__main__":
    main()
