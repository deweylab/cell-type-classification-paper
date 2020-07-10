##########################################################################
#   Run a classifier on a test set
##########################################################################

from optparse import OptionParser
import json
from os.path import join
import dill
from collections import defaultdict
import pandas as pd
import numpy as np
import scipy

import train_model
from common import load_dataset

QUALIFIER_TERMS = set([
    'CL:2000001',   # peripheral blood mononuclear cell
    'CL:0000081',   # blood cell
    'CL:0000080'    # circulating cell
])

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-m", 
        "--model_f", 
        help="Load model from file"
    )
    parser.add_option(
        "-t",
        "--train_dir",
        help="Training dataset directory"
    )
    parser.add_option(
        "-p",
        "--train_params",
        help="Training parameters"
    )
    parser.add_option(
        "-c",
        "--classification_threshold",
        help="Classification score to use as the threshold for a positive classification"
    )
    parser.add_option(
        "-f",
        "--classification_threshold_file",
        help="Path to JSON file mapping each label to its classification threshold for calling a positive classification"
    )
    parser.add_option(
        "-i",
        "--finalizer",
        action='store_true',
        help="Finalizer"
    )
    parser.add_option(
        "-e",
        "--finalizer_features",
        help="Finalizer features"
    )
    parser.add_option(
        "-b",
        "--only_binary",
        action="store_true",
        help="Do not output score or probability files)"
    )
    parser.add_option(
        "-o", 
        "--out_dir", 
        help="Directory in which to write output"
    )
    (options, args) = parser.parse_args()

    test_data_dir = args[0]
    out_dir = options.out_dir

    if options.model_f:
        with open(options.model_f, 'rb') as f:
            mod = dill.load(f)
        test_features = args[1] 
    else:
        assert options.train_dir is not None
        assert options.train_params is not None
        with open(options.train_params, 'r') as f:
            config=json.load(f)
        train_features = config['features']
        algorithm = config['algorithm']
        params = config['params']
        mod = train_model.train_model(
            options.train_dir,
            train_features,
            algorithm, 
            params,
            join(out_dir, 'tmp')
        )        
        test_features = args[1]

    # Load the test data
    r = load_dataset.load_dataset(
        test_data_dir,
        test_features
    )
    the_exps = r[3]
    data_matrix = r[10]
    gene_ids = r[11]

    # Re-order columns of data matrix to be same as expected
    # by the model
    assert frozenset(mod.classifier.features) == frozenset(gene_ids)
    if not tuple(mod.classifier.features) == tuple(gene_ids):
        print('Re-ordering columns of data matrix in accordance with classifier input specification...')
        gene_to_index = {
            gene: i
            for i, gene in enumerate(gene_ids)
        }
        indices = [
            gene_to_index[gene]
            for gene in mod.classifier.features
        ]
        data_matrix = data_matrix[:,indices]
        print('done.')
            

    # TODO REMOVE!
    #all_indices = np.arange(len(the_exps))
    #rand_indices = np.random.choice(all_indices, 1000, replace=False)
    #rand_indices = sorted(rand_indices)
    #the_exps = np.array(the_exps)[rand_indices]
    #data_matrix = data_matrix[rand_indices,:]

    # Apply model
    print('Applying model to test set.')
    confidence_df, score_df = mod.predict(data_matrix, the_exps)
    print('done.')

    # Write output to files
    if not options.only_binary:
        confidence_df.to_csv(
            join(out_dir, 'classification_results.tsv'),
            sep='\t'
        )
        score_df.to_csv(
            join(out_dir, 'classification_scores.tsv'),
            sep='\t'
        )
 
    # Binarize the classifications 
    if options.classification_threshold_file:
        label_to_thresh_df = pd.read_csv(options.classification_threshold_file, sep='\t', index_col=0)
        label_to_thresh = {
            label: label_to_thresh_df.loc[label]['threshold']
            for label in label_to_thresh_df.index
        }
    if options.classification_threshold \
        or options.classification_threshold_file:
        if options.classification_threshold:
            assert options.classification_threshold_file is None
            classif_thresh = float(options.classification_threshold)
            label_to_thresh = {
                label: classif_thresh
                for label in confidence_df.columns
            }
        if options.classification_threshold_file:
            assert options.classification_threshold is None
            label_to_thresh_df = pd.read_csv(options.classification_threshold_file, sep='\t', index_col=0)
            label_to_thresh = {
                label: label_to_thresh_df.loc[label]['threshold']
                for label in label_to_thresh_df.index
            }
        label_graph = mod.classifier.label_graph
        binary_df = _binarize_classifications(
            confidence_df, 
            label_to_thresh, 
            label_graph
        )

        if options.classification_threshold:
            binary_df.to_csv(
                join(out_dir, 'binary_classification_results.thresh_{}.tsv'.format(str(options.classification_threshold))),
                sep='\t'
            )
        elif options.classification_threshold_file:
            binary_df.to_csv(
                join(out_dir, 'binary_classification_results.tsv'),
                sep='\t'
            ) 

    if options.finalizer:
        binary_one_df = _select_one_most_specific(data_matrix, binary_df, options.train_dir, options.finalizer_features)
        binary_one_df.to_csv(
            join(out_dir, 'binary_classification_results_nearest_neigbhor_finalize.tsv'),
            sep='\t'
        )
        

def _select_one_most_specific(data_matrix, binary_results_df, train_dir, feats):
    r = load_dataset.load_dataset(train_dir, feats)
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
    train_data_matrix = r[10]
    gene_ids = r[11]

    # Map each label to its ancestors
    label_to_ancestors = {
        label: label_graph.ancestor_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    # Map each experiment to its predicted terms
    print('Mapping each sample to its predicted labels...')
    exp_to_pred_labels = {
        exp: [
            label
            for label in binary_results_df.columns
            if binary_results_df.loc[exp][label] == 1
        ]
        for exp in binary_results_df.index
    }
    print('Computing the most-specific predicted labels...')
    exp_to_ms_pred_labels = {
        exp: label_graph.most_specific_nodes(set(pred_labels) - QUALIFIER_TERMS)
        for exp, pred_labels in exp_to_pred_labels.items()
    }

    # Map each experiment to its indices
    label_to_study_to_indices = defaultdict(lambda: defaultdict(lambda: []))
    for exp_i, exp in enumerate(the_exps):
        study = exp_to_study[exp]
        for label in exp_to_labels[exp]:
            label_to_study_to_indices[label][study].append(exp_i)
            #label_to_indices[label].append(exp_i)

    # Map each label to its mean vector
    print('Computing a mean expression profile for each training set label...')
    X = np.exp(train_data_matrix) - 1
    vecs = []
    label_to_vecs = defaultdict(lambda: [])
    for label, study_to_indices in label_to_study_to_indices.items():
        print('Computing mean expression profile for label {}...'.format(og.id_to_term[label].name))
        for study, indices in study_to_indices.items():
            X_label = X[indices,:]
            x_agg = np.mean(X_label, axis=0)
            x_agg = np.log(x_agg+1)
            label_to_vecs[label].append(x_agg)

    # Find the closest of the candidates
    exp_to_update_pred = defaultdict(lambda: set())
    for exp, x in zip(binary_results_df.index, data_matrix):
        min_dist = float('inf')
        min_label = None
        cand_labels = exp_to_ms_pred_labels[exp]
        print('Narrowing cell {} down to {} labels: {}'.format(exp, len(cand_labels), [og.id_to_term[x].name for x in cand_labels]))
        for label in exp_to_ms_pred_labels[exp]:
            for vec in label_to_vecs[label]: 
                #dist = scipy.spatial.distance.correlation(
                #    x, 
                #    vec
                #)
                dist = -scipy.stats.spearmanr(x, vec)[0]
                if dist < min_dist:
                    min_dist = dist
                    min_label = label
        print('Selected {}'.format(og.id_to_term[min_label].name))
        print()
        exp_to_update_pred[exp] = label_to_ancestors[min_label] 

    # Create the final dataframe 
    da = []
    for exp in binary_results_df.index:
        row = []
        for label in binary_results_df.columns:
            if label in exp_to_update_pred[exp]:
                row.append(1)
            else:
                row.append(0)
        da.append(row)

    df = pd.DataFrame(
        data=da,
        columns=binary_results_df.columns,
        index=binary_results_df.index
    )
    return df 

    
def _binarize_classifications(confidence_df, label_to_thresh, label_graph):
    print('Binarizing classifications...')
    label_to_descendents = {
        label: label_graph.descendent_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    da = []
    the_labels = sorted(set(confidence_df.columns) & set(label_to_thresh.keys()))
    for exp_i, exp in enumerate(confidence_df.index):
        if (exp_i+1) % 100 == 0:
            print('Processed {} samples.'.format(exp_i+1))
        # Map each label to its classification-score 
        label_to_conf = {
            label: confidence_df.loc[exp][label]
            for label in confidence_df.columns
        }
        # Compute whether each label is over its threshold
        label_to_is_above = {
            label: int(conf > label_to_thresh[label])
            for label, conf in label_to_conf.items()
            if label in the_labels
        }
        label_to_bin= {
            label: is_above
            for label, is_above in label_to_is_above.items()
        }
        # Propagate the negative predictions to all descendents
        for label, over_thresh in label_to_is_above.items():
            if not bool(over_thresh):
                desc_labels = label_to_descendents[label]
                for desc_label in set(desc_labels) & set(label_to_bin.keys()):
                    label_to_bin[desc_label] = int(False)
        da.append([
            label_to_bin[label]
            for label in the_labels
        ])
    df = pd.DataFrame(
        data=da,
        index=confidence_df.index,
        columns=the_labels
    )
    return df

if __name__ == "__main__":
    main()
