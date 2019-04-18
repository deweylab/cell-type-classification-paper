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

sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/recount2")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/learning")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from machine_learning import learners
import load_boiler_plate_data_for_exp_list as lbpdfel
import project_data_on_ontology as pdoo

ALGO_CONFIGS_DIR = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/algo_config"

def main():
    usage = "usage: %prog <config file> <environment dir> <experiment list name>"
    parser = OptionParser(usage)
    parser.add_option("-o", "--out_dir", help="Directory in which to write the pickled model")
    parser.add_option("-r", "--artifacts_parent_dir", help="The directory in which to write temporary files")
    (options, args) = parser.parse_args()

    config_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    out_dir = options.out_dir
    artifacts_parent_dir = options.artifacts_parent_dir 

    print "environment directory: %s" % env_dir
    print "experiment list: %s" % exp_list_name

    # load the configuration
    print "reading configuration from %s..." % config_f 
    with open(config_f, 'r') as f:
        config = json.load(f)
    algo_config_f = config['algo_config_file']
    features = config['features']
    dim_reduct_config_f = None
    if 'dim_reduct_config_file' in config:
        dim_reduct_config_f = config['dim_reduct_config_file']    

    # load the data
    r = lbpdfel.load_everything(
        env_dir, 
        exp_list_name, 
        features
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

    # load classifier configuration
    algo_config_f = join(ALGO_CONFIGS_DIR, algo_config_f)
    print "reading algorithm configuration from %s..." % algo_config_f
    with open(algo_config_f, 'r') as f:
        algo_config = json.load(f)
    print algo_config
    algorithm = algo_config['algorithm']
    classif_name = basename(algo_config_f)[:-5]
    classif_params = algo_config['params']

    # load dimension reduction algorithm
    dim_reduct_name = None
    dim_reduct_params = None
    if dim_reduct_config_f:
        with open(dim_reduct_config_f, 'r') as f:
            dim_reduct_config = json.load(f)
            dim_reduct_name = dim_reduct_config['algorithm']
            dim_reduct_params = dim_reduct_config['params']
    
    # compute the artifact directory
    artifact_dir = join(artifacts_parent_dir, "%s.%s" % (
        exp_list_name.replace(".", "_"),
        classif_name.replace(".", "_")
    ))

    # train the model
    model = learners.train_learner(
        algorithm,
        classif_params,
        data_matrix,
        the_exps,
        exp_to_labels,
        label_graph,
        dim_reductor_name=dim_reduct_name,
        dim_reductor_params=dim_reduct_params,
        item_to_group=exp_to_study,
        artifact_dir=artifact_dir,
        feat_names=gene_names
    )
    #stds = np.std(data_matrix, axis=0)
    #label_to_std_coefs = {}
    #for label, classif in model.classifier.label_to_classifier.iteritems():
    #    std_coefs = np.multiply(classif.coef_[0], stds)
    #    label_to_std_coefs[pdoo.convert_node_to_str(label)] = list(std_coefs)

    print "pickling the model..."
    #out_f = join(
    #    out_dir, 
    #    "%s.%s.pickle" % (
    #        exp_list_name, 
    #        classif_name
    #    )
    #)
    out_f = join(out_dir, 'model.pickle')
    with open(out_f, 'w') as f:
        cPickle.dump(model, f)
    print "done."

    #print "writing standardized coefficients..."
    #out_f = join(
    #    out_dir,
    #    "%s.%s.std_coeffs.json" % (
    #        exp_list_name,
    #        classif_name
    #    )
    #)
    #with open(out_f, 'w') as f:
    #    f.write(json.dumps(label_to_std_coefs))
    #print "done."

   

if __name__ == "__main__":
    main()
