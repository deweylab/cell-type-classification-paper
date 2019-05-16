from optparse import OptionParser
import json
from collections import defaultdict
import os
from os.path import join
import sys
import random
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy as np
import math

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import load_boiler_plate_data_for_exp_list as lbpdfel
import project_data_on_ontology as pdoo

MIN_EXPS = 10
N_NEGATIVE_STUDIES = 30
N_TRAIN_SETS_PER_STUDY = 30

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write output")
    (options, args) = parser.parse_args()

    env_dir = args[0]
    exp_list_name = args[1]
    out_dir = options.out_dir

    r = lbpdfel.load_everything(
        env_dir,
        exp_list_name,
        'kallisto_gene_log_cpm'
        #'random'
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

    label_to_studies = defaultdict(lambda: set())
    for exp, labels in exp_to_labels.iteritems():
        study = exp_to_study[exp]
        for label in labels:
            label_to_studies[label].add(study)
    label_to_studies = dict(label_to_studies)

    skip_labels = set([
        frozenset(["CL:0000548"]),  # animal cell
        frozenset(["CL:0000255"]),  # eukaryotic cell
        frozenset(["CL:0000010"]),  # cultured cell
        frozenset(["CL:0000578"]),  # experimentally modified cell in vitro
        frozenset(["CL:0001034"])   # cell in vitro
    ])

    # Map each label to its experiments
    label_to_exps = defaultdict(lambda: set())
    for exp, labels in exp_to_labels.iteritems():
        for label in labels:
            label_to_exps[label].add(exp)
    label_to_exps = dict(label_to_exps)

    # Generate the training sets for each label
    label_to_pos_exps, label_to_neg_exps = training_sets(
        the_exps, 
        exp_to_labels, 
        label_to_exps, 
        label_graph
    )

    # Map each label to a dictionary mapping each study to all
    # samples in that label and study. 
    label_to_pos_study_to_exps = {}
    label_to_neg_study_to_exps = {}
    valid_labels = set()
    for label, pos_exps in label_to_pos_exps.iteritems():
        if label in skip_labels:
            continue 
        pos_study_to_exps = defaultdict(lambda: set())
        for exp in pos_exps:
            study = exp_to_study[exp]
            pos_study_to_exps[study].add(exp)
        pos_study_to_exps = dict(pos_study_to_exps)

        # We only want to use studies with a certain
        # number of samples
        pos_study_to_exps = {
            study: exps 
            for study, exps in pos_study_to_exps.iteritems()
            if len(exps) > MIN_EXPS
        }
        label_to_pos_study_to_exps[label] = pos_study_to_exps
        if len(pos_study_to_exps) >= 3:
            valid_labels.add(label)
            print "Label %s has %d studies with at least %d experiments" % (
                label_to_name[label], 
                len(pos_study_to_exps), 
                MIN_EXPS
            )

    valid_labels = label_graph.most_specific_nodes(valid_labels)
    print "The most specific labels I can perform the experiments on are: %s" % valid_labels

    # Map each label to a dictioanry mapping each study to 
    # all negative samples for that study
    for label, neg_exps in label_to_neg_exps.iteritems():
        if label in skip_labels:
            continue
        neg_study_to_exps = defaultdict(lambda: set())
        for exp in neg_exps:
            study = exp_to_study[exp]
            neg_study_to_exps[study].add(exp)
        neg_study_to_exps = dict(neg_study_to_exps)
        label_to_neg_study_to_exps[label] = neg_study_to_exps
    
    label_to_homo_recalls = defaultdict(lambda: [])
    label_to_hetero_recalls = defaultdict(lambda: [])
    label_to_homo_true_neg_rates = defaultdict(lambda: [])
    label_to_hetero_true_neg_rates = defaultdict(lambda: [])
    label_to_held_out_study = defaultdict(lambda: [])
    label_to_homo_avg_precs = defaultdict(lambda: [])
    label_to_hetero_avg_precs = defaultdict(lambda: [])
    label_to_homo_aucs = defaultdict(lambda: [])
    label_to_hetero_aucs = defaultdict(lambda: [])
    for curr_label in valid_labels:
        pos_study_to_exps = label_to_pos_study_to_exps[curr_label]
        print 
        print "Performing experiment on label %s" % label_to_name[curr_label] 
        for held_out_study, held_out_exps in pos_study_to_exps.iteritems():
            held_out_exps = list(pos_study_to_exps[held_out_study])

            print "Holding out study %s" % held_out_study
            
            # Compute the set of studies we can draw from for each  
            # training set
            train_pos_studies = set(pos_study_to_exps.keys()) - set([held_out_study])
            print "There are %s candidate studies that we can train on" % len(train_pos_studies)

            # To ensure that the same amount of data is used in each
            # training set, we must compute the minimum number of
            # experiments in the candidate studies
            min_num_exps = min([
                len(pos_study_to_exps[study])
                for study in train_pos_studies
            ])
            print "The minimum number of experiments across the candidate studies is %d" % min_num_exps

            # For each study, create a set of homogenous training sets
            # by sampling N samples from that study.
            study_to_homo_train_sets = {}
            for study in train_pos_studies:
                study_train_sets = []
                for i in range(N_TRAIN_SETS_PER_STUDY):
                    train_set = list(random.sample(pos_study_to_exps[study], min_num_exps))
                    study_train_sets.append(train_set)
                study_to_homo_train_sets[study] = study_train_sets
            
            # Create sets of heterogeneous training sets
            hetero_train_sets = []
            per_study_size = float(min_num_exps) / len(train_pos_studies) 
            print "When creating heterogenous training sets, we draw %f experiments from each study" % per_study_size
            if per_study_size < 1.0:
                sample_from_studies = random.sample(train_pos_studies, min_num_exps)
                per_study_size = 1
                print "Oh no, the number of samples to draw is less than 1. So we are only sampling one from %d random studies" % len(sample_from_studies)
                sample_one_from_studies = []
            else:
                sample_from_studies = list(train_pos_studies)
                per_study_size = int(math.floor(per_study_size))
                if per_study_size * len(train_pos_studies) < min_num_exps:
                    sample_more = min_num_exps - (per_study_size * len(train_pos_studies))
                    print "Uh oh. If we only sample %d from each of %d studies, we won't have enough. So we need more %d more" % (
                        per_study_size, 
                        len(train_pos_studies), 
                        sample_more
                    )
                    sample_one_from_studies = list(random.sample(train_pos_studies, sample_more))
                    print "Sampling one experiment from each of %s" % sample_one_from_studies
                else:
                    sample_one_from_studies = []
            for i in range(N_TRAIN_SETS_PER_STUDY):
                curr_train_set = set()
                for study in sample_from_studies:
                    curr_train_set.update(
                        random.sample(pos_study_to_exps[study], per_study_size)
                    )
                for study in sample_one_from_studies:
                    candidate_exps = set(pos_study_to_exps[study])-curr_train_set
                    curr_train_set.update(
                        random.sample(candidate_exps, 1)
                    )
                hetero_train_sets.append(list(curr_train_set))


            # Compute the negative examples to place in the test set.
            # Make sure these can never appear in the training set. 
            cand_neg_studies = set(label_to_neg_study_to_exps[curr_label].keys()) \
                - set(label_to_pos_study_to_exps[curr_label].keys())
            neg_test_studies = set(random.sample(
                cand_neg_studies, 
                N_NEGATIVE_STUDIES
            ))
            neg_test_exps = [] 
            for study in neg_test_studies:
                exp = random.sample(label_to_neg_study_to_exps[curr_label][study], 1)[0]
                neg_test_exps.append(exp) 

            # Compute the set of experiments to use for the negative
            # examples in the training set, these should not belong to
            # any study in the test set
            neg_train_exps = set()
            for study, exps in label_to_neg_study_to_exps[curr_label].iteritems():
                if study != held_out_study and study not in neg_test_studies:
                    neg_train_exps.update(random.sample(exps, 1)) # To make training sets more balanced
            neg_train_exps = list(neg_train_exps)
            
            held_out_pos_vecs = [
                data_matrix[exp_to_index[exp]]
                for exp in held_out_exps
            ]
            held_out_neg_vecs = [
                data_matrix[exp_to_index[exp]]
                for exp in neg_test_exps
            ]
            neg_train_set_vecs = [
                data_matrix[exp_to_index[exp]]
                for exp in neg_train_exps
            ]
            neg_train_set_binlabels = [
                0
                for exp in neg_train_exps
            ]
            print "There are %d negative examples." % len(neg_train_set_vecs)

            # Train a classifier on each training set
            print "Training on the homogeneous training sets..."
            study_to_homo_training_set_models = {}
            for study, train_sets in study_to_homo_train_sets.iteritems():
                models = []
                for train_set in train_sets:
                    assert isinstance(train_set, list)
                    assert len(set(train_set) & set(neg_train_exps)) == 0
                    assert len(set(train_set) & set(held_out_exps)) == 0
                    assert len(set(train_set) & set(neg_test_exps)) == 0
                    train_set_vecs = [
                        data_matrix[exp_to_index[exp]]
                        for exp in train_set
                    ]
                    train_set_binlabels = [
                        1
                        for exp in train_set
                    ]
                    print "%d positive examples in training set" % len(train_set_binlabels)
                    model = LogisticRegression(penalty='l2', tol=1e-9)
                    model.fit(
                        train_set_vecs + neg_train_set_vecs,
                        train_set_binlabels + neg_train_set_binlabels
                    )
                    models.append(model)
                study_to_homo_training_set_models[study] = models

            print "Training on the heterogeneous training sets..."
            hetero_train_set_models = []
            for train_set in hetero_train_sets:
                assert isinstance(train_set, list)
                assert len(set(train_set) & set(neg_train_exps)) == 0
                assert len(set(train_set) & set(held_out_exps)) == 0
                assert len(set(train_set) & set(neg_test_exps)) == 0
                train_set_vecs = [
                    data_matrix[exp_to_index[exp]]
                    for exp in train_set
                ]
                train_set_binlabels = [
                    1
                    for exp in train_set
                ]
                print "%d positive examples in training set" % len(train_set_binlabels)
                model = LogisticRegression(penalty='l2', tol=1e-9)
                model.fit(
                    train_set_vecs + neg_train_set_vecs,
                    train_set_binlabels + neg_train_set_binlabels
                )
                hetero_train_set_models.append(model)

            # Test each classifier on the held out study and
            # and compute the metrics
            study_to_homo_train_set_recalls = {}
            study_to_homo_train_set_true_neg_rates = {}
            study_to_homo_train_set_avg_precs = {}
            study_to_homo_train_set_aucs = {}
            print "Examining the homogeneously trained models."
            for study, models in study_to_homo_training_set_models.iteritems():
                recalls = []
                true_neg_rates = []
                avg_precs = []
                aucs = []
                for model in models:
                    # determine the index of the positive vs. negative
                    # prediction score
                    for ind, clss in enumerate(model.classes_):
                        if clss == 1:
                            pos_index = ind
                        elif clss == 0:
                            neg_index = ind

                    # Compute the predictions and prediction scores
                    pos_preds = model.predict(held_out_pos_vecs)
                    neg_preds = model.predict(held_out_neg_vecs)
                    pos_scores = [
                        x[pos_index]
                        for x in model.predict_proba(held_out_pos_vecs)
                    ]
                    neg_scores = [
                        x[pos_index]
                        for x in model.predict_proba(held_out_neg_vecs)
                    ]
                    all_scores = pos_scores + neg_scores
                    all_true_labels = [
                        True
                        for x in pos_scores
                    ]
                    all_true_labels += [
                        False
                        for x in neg_scores
                    ]

                    # Since a positive label is 1 and negative label is
                    # 0, we can compute the recall by summing over the 
                    # predictions
                    recall = float(sum(pos_preds)) / len(pos_preds)
                    true_neg_rate = float(len(neg_preds) - sum(neg_preds)) / len(neg_preds)
                    avg_prec = average_precision_score(all_true_labels, all_scores)
                    auc = roc_auc_score(all_true_labels, all_scores)
                    
                    recalls.append(recall)
                    true_neg_rates.append(true_neg_rate)
                    avg_precs.append(avg_prec)
                    aucs.append(auc)
                study_to_homo_train_set_recalls[study] = recalls
                study_to_homo_train_set_true_neg_rates[study] = true_neg_rates
                study_to_homo_train_set_avg_precs[study] = avg_precs
                study_to_homo_train_set_aucs[study] = aucs

            hetero_train_set_recalls = []
            hetero_train_set_true_neg_rates = []
            hetero_train_set_avg_precs = []
            hetero_train_set_aucs = []
            print "Examining the heterogeneously trained models."
            for model in hetero_train_set_models:
                # determine the index of the positive vs. negative
                # prediction score
                for ind, clss in enumerate(model.classes_):
                    if clss == 1:
                        pos_index = ind
                    elif clss == 0:
                        neg_index = ind

                # Compute the predictions and prediction scores
                pos_preds = model.predict(held_out_pos_vecs)
                neg_preds = model.predict(held_out_neg_vecs)
                pos_scores = [
                    x[pos_index]
                    for x in model.predict_proba(held_out_pos_vecs)
                ]
                neg_scores = [
                    x[pos_index]
                    for x in model.predict_proba(held_out_neg_vecs)
                ]
                all_scores = pos_scores + neg_scores
                all_true_labels = [
                    True
                    for x in pos_scores
                ]
                all_true_labels += [
                    False
                    for x in neg_scores
                ]

                # Since a positive label is 1 and negative label is
                # 0, we can compute the recall by summing over the 
                # predictions
                recall = float(sum(pos_preds)) / len(pos_preds)
                true_neg_rate = float(len(neg_preds) - sum(neg_preds)) / len(neg_preds)
                avg_prec = average_precision_score(all_true_labels, all_scores)
                auc = roc_auc_score(all_true_labels, all_scores) 
                print "The AUC is: %f" % auc

                hetero_train_set_recalls.append(recall)
                hetero_train_set_true_neg_rates.append(true_neg_rate) 
                hetero_train_set_avg_precs.append(avg_prec)
                hetero_train_set_aucs.append(auc)

            label_to_homo_recalls[curr_label].append(study_to_homo_train_set_recalls)
            label_to_hetero_recalls[curr_label].append(hetero_train_set_recalls)
            label_to_homo_true_neg_rates[curr_label].append(study_to_homo_train_set_true_neg_rates) 
            label_to_hetero_true_neg_rates[curr_label].append(hetero_train_set_true_neg_rates)
            label_to_homo_avg_precs[curr_label].append(study_to_homo_train_set_avg_precs)
            label_to_hetero_avg_precs[curr_label].append(hetero_train_set_avg_precs)
            label_to_homo_aucs[curr_label].append(study_to_homo_train_set_aucs)
            label_to_hetero_aucs[curr_label].append(hetero_train_set_aucs)
            label_to_held_out_study[curr_label].append(held_out_study)
            print "Homogeneous training set recalls: %s" % study_to_homo_train_set_recalls
            print "Heterogeneous training set recalls: %s" % hetero_train_set_recalls
            print "Homogeneous training set true negative rates: %s" % study_to_homo_train_set_true_neg_rates
            print "Heterogeneous training set true negative rates: %s" % hetero_train_set_true_neg_rates
            print "Homogenous training set avg precs: %s" % study_to_homo_train_set_avg_precs
            print "Heterogeneous training set avg precs: %s" % hetero_train_set_avg_precs      
            print "Homogenous training set AUCs: %s" % study_to_homo_train_set_aucs
            print "Heterogeneous training set AUCs: %s" % hetero_train_set_aucs 
     
    out_f = join(
        out_dir,
        "homo_vs_hetero_training_results.min_%s.json" % MIN_EXPS
    )
    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "homogeneous_recalls": {
                    pdoo.convert_node_to_str(label): recalls
                    for label, recalls in label_to_homo_recalls.iteritems()
                },
                "heterogeneous_recalls": {
                    pdoo.convert_node_to_str(label): recalls
                    for label, recalls in label_to_hetero_recalls.iteritems()
                },
                "homogeneous_true_negative_rates": {
                    pdoo.convert_node_to_str(label): tnrs
                    for label, tnrs in label_to_homo_true_neg_rates.iteritems()
                },
                "heterogeneous_true_negative_rates": {
                    pdoo.convert_node_to_str(label): tnrs
                    for label, tnrs in label_to_hetero_true_neg_rates.iteritems()
                },
                "homogeneous_avg_precs": {
                    pdoo.convert_node_to_str(label): avg_precs
                    for label, avg_precs in label_to_homo_avg_precs.iteritems()
                },
                "heterogeneous_avg_precs": {
                    pdoo.convert_node_to_str(label): avg_precs
                    for label, avg_precs in label_to_hetero_avg_precs.iteritems()
                },
                "homogeneous_aucs": {
                    pdoo.convert_node_to_str(label): aucs
                    for label, aucs in label_to_homo_aucs.iteritems()
                },
                "heterogeneous_aucs": {
                    pdoo.convert_node_to_str(label): aucs
                    for label, aucs in label_to_hetero_aucs.iteritems()
                },
                "held_out_studies": {
                    pdoo.convert_node_to_str(label): study
                    for label, study in label_to_held_out_study.iteritems()
                }
            },
            indent=True
        ))


def training_sets(all_exps, exp_to_labels, label_to_exps, label_graph):
    # Compute the positive items for each label
    # This set consists of all items labelled with a 
    # descendent of the label
    print "Computing positive labels..."
    label_to_pos_exps = {}
    for label in label_to_exps:
        pos_exps = label_to_exps[label].copy()
        desc_labels = label_graph.descendent_nodes(label)
        for desc_label in desc_labels:
            if desc_label in label_to_exps:
                pos_exps.update(label_to_exps[desc_label])
        label_to_pos_exps[label] = pos_exps

    # Compute the negative items for each label
    # This set consists of all items that are not labelled
    # with a descendant of the label, but are also not labelled
    # with an ancestor of the label (can include siblings).
    print "Computing negative labels..."
    label_to_neg_exps = {}
    for label in label_to_exps:
        neg_exps = set()
        anc_labels = label_graph.ancestor_nodes(label)
        candidate_exps = set(all_exps) - set(label_to_pos_exps[label])
        for exp in candidate_exps:
            ms_exp_labels = label_graph.most_specific_nodes(
                exp_to_labels[exp]
            )
            ms_exp_labels = set(ms_exp_labels)
            if len(ms_exp_labels & anc_labels) == 0:
                neg_exps.add(exp)
        label_to_neg_exps[label] = list(neg_exps)
    return label_to_pos_exps, label_to_neg_exps

if __name__ == "__main__":
    main()
