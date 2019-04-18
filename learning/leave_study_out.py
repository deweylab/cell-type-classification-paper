#########################################################################
#   A function for performing a single 'leave-study-out' train-then-test
#   iteration. That is, given a study and dataset, train the classifier 
#   on all experiments in the dataset that aren't in the target study. 
#   Run the classifier on all experiments in the target study.
#########################################################################

import sys
import os
from os.path import join

from optparse import OptionParser
import json
import collections
from collections import defaultdict, Counter
import numpy as np
import subprocess

sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/recount2")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/learning")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

from machine_learning import learners
import project_data_on_ontology as pdoo
import experiment
import polya_total_rna_feature_correction as ptrfc
       
def leave_study_out(
        leave_out_study,
        test_exps,
        data_matrix,
        the_exps,
        study_to_exps,
        exp_to_study,
        exp_to_terms,
        label_graph,
        exp_to_labels,
        og,
        classif_name,
        classif_params,
        collapse_ontology,
        gene_names,
        dim_reduct_name=None,
        dim_reduct_params=None,
        artifacts_parent_dir=None,
        verbose=False,
        normalize_polyA_total=False,
        exp_to_tags=None
    ):

    the_studies = set(study_to_exps.keys())
    exp_to_index = {
        x: i 
        for i, x in enumerate(the_exps)
    }
    exp_to_label_str_to_conf = {}
    exp_to_label_str_to_score = {}
    # Gather fold's training data
    fold_train_studies = set([
        x
        for x in the_studies
        if x != leave_out_study
    ])
    train_vecs = []
    train_exps = []
    for study in fold_train_studies:
        for exp in study_to_exps[study]:
            train_vecs.append(
                data_matrix[exp_to_index[exp]]
            )
            train_exps.append(exp)
    train_vecs = np.asarray(train_vecs)

    # Compute the fold's label_graph
    train_exps_set = set(train_exps)
    train_exp_to_terms = {
        exp: terms
        for exp, terms in exp_to_terms.iteritems()
        if exp in train_exps_set
    }

    if collapse_ontology:
        fold_label_graph, fold_exp_to_labels = pdoo.collapsed_ontology_graph(train_exp_to_terms)
    else:
        fold_label_graph, fold_exp_to_labels = pdoo.full_ontology_graph(og, train_exp_to_terms)

    # Gather fold's test data
    test_vecs = [
        data_matrix[exp_to_index[exp]]
        for exp in test_exps
    ]

    if artifacts_parent_dir:
        artifact_dir = join(
            artifacts_parent_dir,
            "artifacts_%s" % leave_out_study
        )
        run_cmd([
            "mkdir",
            "-p",
            artifact_dir
        ])
    else:
        artifact_dir = None

    # Normalize the features between Poly-A and tota RNA-seq
    # samples
    if normalize_polyA_total:
        train_exp_to_study = {
            exp: study
            for exp, study in exp_to_study.iteritems()
            if exp in train_exps_set
        }
        train_exp_to_tags = {
            exp: exp_to_tags[exp]
            for exp in train_exps
        }
        train_exp_to_labels = {
            exp: exp_to_labels[exp]
            for exp in train_exps_set
        }

        # Train the RNA-selection classifier
        selection_classif = ptrfc.train_rna_selection_classifier(
            train_exps, 
            train_vecs, 
            train_exp_to_tags, 
            train_exp_to_study, 
            train_exp_to_labels, 
            fold_label_graph
        )

        # Normalize the features in the training set
        train_vecs = np.copy(train_vecs)
        train_vecs = ptrfc.normalize_features(
            train_exps,
            train_vecs,
            train_exp_to_tags,
            gene_names,
            selection_classif
        )

        # Run the classifier on the test set and 
        # normalize according to the results
        test_vecs = np.copy(test_vecs)
        test_exp_to_tags = defaultdict(lambda: set())
        test_vecs = ptrfc.normalize_features(
            test_exps,
            test_vecs,
            test_exp_to_tags,
            gene_names,
            selection_classif
        )


    model = learners.train_learner(
        classif_name,
        classif_params,
        train_vecs,
        train_exps,
        fold_exp_to_labels,
        fold_label_graph,
        dim_reductor_name=dim_reduct_name,
        dim_reductor_params=dim_reduct_params,
        item_to_group=exp_to_study,
        artifact_dir=artifact_dir,
        verbose=verbose
    )

    # Returns a list of dictionaries. Each dictionary corresponds
    # to a sample and maps a label/node to its confidence of being
    # assigned
    node_to_confidences, node_to_classif_scores = model.predict(test_vecs)
    #print "NODE TO CONFIDENCES: %s" % node_to_confidences
    #print "NODE TO CLASSIFIER SCORES: %s" % node_to_classif_scores
    node_to_confidences = experiment.convert_to_global_labels(
        node_to_confidences, 
        label_graph
    )
   
    # TODO this should be removed at some point. I'm currently using it
    # intermittently to debug the BNC Gibbs sampling algorithm
    #with open(join(artifact_dir, "test_experiments_order.json"), 'w') as f:
    #    f.write(json.dumps(test_exps, indent=True))

    for test_exp, node_to_confidence in zip(test_exps, node_to_confidences):
        exp_to_label_str_to_conf[test_exp] = {
            pdoo.convert_node_to_str(node): conf
            for node, conf in node_to_confidence.iteritems()
        }

    node_to_classif_scores = experiment.convert_to_global_labels(
        node_to_classif_scores,
        label_graph
    )
    for test_exp, node_to_classif_score in zip(test_exps, node_to_classif_scores):
        exp_to_label_str_to_score[test_exp] = {
            pdoo.convert_node_to_str(node): score
            for node, score in node_to_classif_score.iteritems()
        }

    return exp_to_label_str_to_conf, exp_to_label_str_to_score


def run_cmd(cmd_toks):
    print " ".join(cmd_toks)
    return subprocess.Popen(cmd_toks)


if __name__ == "__main__":
    main()
