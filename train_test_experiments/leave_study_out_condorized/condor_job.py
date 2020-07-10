#############################################################
#   This is the script that is kicked off by the Condor
#   executable
#############################################################
from optparse import OptionParser
from os.path import join
import pandas as pd
import json
import numpy as np

from graph_lib import graph
from models import model
from common import the_ontology
from common import load_dataset
from common import ontology_utils

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    parser.add_option("-a", "--algo_config_dir", help="The directory where all the classifier configurations are stored")
    parser.add_option("-r", "--artifacts_dir", help="The directory in which to write temporary files")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    config_f = args[0]
    dataset_dir = args[1]
    fold_f = args[2]
    out_f = options.out_file

    # Load training configuration
    with open(config_f, 'r') as f:
        training_config = json.load(f)
    params = training_config['params']
    features = training_config['features']
    algorithm = training_config['algorithm']
    preprocessors = None
    preprocessor_params = None
    if 'preprocessors' in training_config:
        assert 'preprocessor_params' in training_config
        preprocessors = training_config['preprocessors']
        preprocessor_params = training_config['preprocessor_params']

    # Load the dataset
    r = load_dataset.load_dataset(dataset_dir, features)
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
    gene_ids = r[11]

    # Load the fold's study and training/test sets
    with open(fold_f, 'r') as f:
       fold = json.load(f)
    held_exps = fold['experiments']
    held_study = fold['study']
    fold_exps = set(the_exps) - set(held_exps)

    # Map the fold's training experiments to their
    # label sets
    fold_exp_to_labels = {
        exp: exp_to_labels[exp]
        for exp in fold_exps
    }

    # Build the ontology-graph spanning this
    # fold's training set
    og = the_ontology.the_ontology()
    all_labels = set()
    for labels in fold_exp_to_labels.values():
        all_labels.update(labels)
    fold_label_graph = ontology_utils.ontology_subgraph_spanning_terms(
        all_labels,
        og
    )
    fold_label_graph = graph.transitive_reduction_on_dag(
        fold_label_graph
    )

    print('Training model...')
    fold_data_df = pd.DataFrame(
        data=data_matrix,
        index=the_exps,
        columns=gene_ids
    )
    fold_data_df = fold_data_df.loc[fold_exps]
    fold_data_matrix = np.array(fold_data_df)
    out_dir = '.'
    mod = model.train_model(
        algorithm,
        params,
        fold_data_matrix,
        fold_exps,
        fold_exp_to_labels,
        fold_label_graph,
        item_to_group=None,
        tmp_dir=join(out_dir, 'tmp'),
        features=gene_ids,
        preprocessor_names=preprocessors,
        preprocessor_params=preprocessor_params
    )
    print('done.')

    # Apply model on held-out data
    print('Applying model to test set.')
    held_data_df = pd.DataFrame(
        data=data_matrix,
        index=the_exps,
        columns=gene_ids
    )
    held_data_df = held_data_df.loc[held_exps]
    held_data_matrix = np.array(held_data_df)
    confidence_df, score_df = mod.predict(
        held_data_matrix, 
        held_data_df.index
    )
    print('done.')

    # Write output
    confidence_df.to_csv(out_f, sep='\t')




 

if __name__ == "__main__":
    main()
