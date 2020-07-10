#########################################################################
#   Train a hiearchical classifier on an experiment list and pickle the
#   model.
#########################################################################

import sys
import os
from os.path import join, basename
from optparse import OptionParser
import json
import collections
from collections import defaultdict, Counter
import numpy as np
import dill

from models import model
from common import load_dataset

def main():
    usage = "usage: %prog <configuration_file> <dataset_directory>"
    parser = OptionParser(usage)
    parser.add_option(
        "-m",
        "--pretrained_ensemble",
        help="Path to a dilled pre-trained ensemble of classifiers"
    )
    parser.add_option(
        "-o", 
        "--out_dir", 
        help="Directory in which to write the model"
    )
    (options, args) = parser.parse_args()

    config_f = args[0]
    dataset_dir = args[1]
    out_dir = options.out_dir
    if options.pretrained_ensemble:
        pretrained_ensemble_f = options.pretrained_ensemble
    else:
        pretrained_ensemble_f = None

    # Load the configuration
    print("Reading configuration from {}.".format(config_f)) 
    with open(config_f, 'r') as f:
        config = json.load(f)
    features = config['features']
    algorithm = config['algorithm']
    params = config['params']
    preprocessors = None
    preprocessor_params = None
    if 'preprocessors' in config:
        assert 'preprocessor_params' in config
        preprocessors = config['preprocessors']
        preprocessor_params = config['preprocessor_params']

    # Train model
    mod = train_model(
        dataset_dir, 
        features, 
        algorithm, 
        params,
        join(out_dir, 'tmp'),
        preprocessor_names=preprocessors,
        preprocessor_params=preprocessor_params,
        model_dependency=pretrained_ensemble_f
    )

    print("Dumping the model with dill...")
    out_f = join(out_dir, 'model.dill')
    with open(out_f, 'wb') as f:
        dill.dump(mod, f)
    print("done.")

def train_model(
        dataset_dir, 
        features, 
        algorithm, 
        params, 
        tmp_dir, 
        model_dependency=None, 
        preprocessor_names=None,
        preprocessor_params=None
    ):
    # Load the data
    r = load_dataset.load_dataset(
        dataset_dir,
        features
    )
    og = r[0]
    label_graph = r[1]
    label_to_name = r[2]
    the_exps = r[3]
    exp_to_index = r[4]
    exp_to_labels = r[5]
    exp_to_tags = r[6]
    exp_to_study = r[7]
    study_to_exps = r[8]
    exp_to_ms_labels = r[9]
    data_matrix = r[10]
    gene_names = r[11]

    # Train the classifier
    print('Training model: {}'.format(algorithm))
    print('Parameters:\n{}'.format(json.dumps(params, indent=4)))
    if preprocessor_names is not None:
        print('Preprocessing data with: {}'.format(preprocessor_names))
        print('Parameters:\n{}'.format(json.dumps(preprocessor_params, indent=4)))
    mod = model.train_model(
        algorithm,
        params,
        data_matrix,
        the_exps,
        exp_to_labels,
        label_graph,
        item_to_group=exp_to_study,
        tmp_dir=tmp_dir,
        features=gene_names,
        model_dependency=model_dependency,
        preprocessor_names=preprocessor_names,
        preprocessor_params=preprocessor_params
    )
    print('done.')
    return mod



if __name__ == "__main__":
    main()
