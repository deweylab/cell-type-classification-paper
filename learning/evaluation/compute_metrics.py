###########################################################################################
#   Evaluate the output of a leave-study-out cross-validation experiment. More
#   specifically, this script analyzes confidence score outputs.
###########################################################################################

import matplotlib as mpl
mpl.use('Agg')

import os
from os.path import join
import sys

import random
import json
from optparse import OptionParser
import collections
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt
import pandas
import sklearn
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import numpy as np

sys.path.append("/ua/mnbernstein/projects/vis_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import project_data_on_ontology as pdoo
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import vis_lib

def main():
    scores = [-1.0, -0.5, -0.5, 0.0, 0.5, 1.0]
    assigneds  = [False, False, True, True, False, True]
    print _precision_recall_curve(assigneds, scores)


def convert_keys_to_label_strs(label_to_metric, og):
    return {
        pdoo.convert_node_to_name(k, og): v
        for k,v in label_to_metric.iteritems()
    }


def data_metrics(
        samples,
        sample_to_labels,
        sample_to_info
    ):
    """
    Compute metrics on the dataset independently of any predictions
    made by a classifier.
    """
    label_to_studies = defaultdict(lambda: set())
    for sample in samples:
        study = sample_to_info[sample]['study_accession']
        for label in sample_to_labels[sample]:
            label_to_studies[label].add(study)
    label_to_num_studies = {
        k:len(v) 
        for k,v in label_to_studies.iteritems()
    }
    return label_to_num_studies

def _compute_label_to_item_assignments(
        item_to_label_to_conf,
        item_to_labels,
        restrict_to_labels=None        
    ):
    label_to_assigneds = defaultdict(lambda: [])
    label_to_scores = defaultdict(lambda: [])
    item_order = []
    for item, label_to_conf in item_to_label_to_conf.iteritems():
        item_order.append(item)
        for label, conf in label_to_conf.iteritems():
            if restrict_to_labels and label not in restrict_to_labels:
                continue
            label_to_assigneds[label].append(
                label in item_to_labels[item]
            )
            label_to_scores[label].append(conf)
    label_to_assigneds = dict(label_to_assigneds)
    return label_to_assigneds, label_to_scores, item_order 

def _compute_per_label_metrics(
        all_labels,
        label_to_assigneds,
        label_to_scores,
        restrict_to_labels=None
    ):
    label_to_auc = {}
    label_to_avg_precision = {}
    label_to_pr_curve = {}
    for label in all_labels:
        if restrict_to_labels and label not in restrict_to_labels:
            continue
        assigneds = label_to_assigneds[label]
        scores = label_to_scores[label]
        assert len(assigneds) > 0
        if len(frozenset(assigneds)) == 1 and list(frozenset(assigneds))[0] == True:
            print "WARNING! All samples are assigned label %s" %  label
            continue
        elif len(frozenset(assigneds)) == 1 and list(frozenset(assigneds))[0] == False:
            print "ERROR!  All samples are unassigned label %s" %  label
            continue
        try:
            auc = metrics.roc_auc_score(assigneds, scores)
        except:
            print "Unable to compute AUC"
            auc = -1.0
        precisions, recalls = _precision_recall_curve(assigneds, scores)
        avg_precision = _average_precision(precisions, recalls)
        if len(precisions) == 2:
            print "Label %s has two precisions: %s" % (label, precisions)

        label_to_auc[label] = auc
        label_to_avg_precision[label] = avg_precision
        label_to_pr_curve[label] = (precisions, recalls)
    return label_to_auc, label_to_avg_precision, label_to_pr_curve


def compute_metrics(
        item_to_label_to_conf, 
        item_to_labels, 
        label_graph,
        restrict_to_labels=None,
        og=None
    ):
    """
    Compute a variety of metrics on the performance of the classifier.
    Args:
        item_to_label_to_conf: a dictionary mapping each item to 
            a dictionary that maps each label to the confidence
            score for that label for that item
        restrict_to_labels: the set of labels that we want to compute
            metrics for. Ignore all labels that are not in this set
    """
    labels = set(label_graph.get_all_nodes())
    label_to_assigneds, label_to_scores, item_order = _compute_label_to_item_assignments(
        item_to_label_to_conf,
        item_to_labels,
        restrict_to_labels=restrict_to_labels
    )
    #for label, assigneds in label_to_assigneds.iteritems():
    #    if len(frozenset(assigneds)) == 1:
    #        print "Huh?? Label %s [%s] has only one assignment value... %s. (%d total samples)" %  (
    #            label, pdoo.convert_node_to_name(label, og), frozenset(assigneds), len(assigneds))

    label_to_auc, label_to_avg_precision, label_to_pr_curve = _compute_per_label_metrics(
        labels,
        label_to_assigneds,
        label_to_scores,
        restrict_to_labels=restrict_to_labels
    )

    # Compute the pan-dataset precision-recall curve
    global_assigneds = []  
    global_scores = []
    for item, label_to_conf in item_to_label_to_conf.iteritems():
        for label, conf in label_to_conf.iteritems():
            if restrict_to_labels and label not in restrict_to_labels:
                continue
            global_assigneds.append(
                label in item_to_labels[item]
            )
            global_scores.append(conf)
    global_precisions, global_recalls = _precision_recall_curve(global_assigneds, global_scores)
    global_pr_curve = (global_precisions, global_recalls)
    global_avg_precision = _average_precision(global_precisions, global_recalls)

    return (
        label_to_auc, 
        label_to_avg_precision, 
        label_to_pr_curve,
        global_pr_curve,
        global_avg_precision
    )


def compute_conservative_metrics(
        item_to_label_to_conf,
        item_to_labels,
        label_graph,
        label_to_name,
        restrict_to_labels=None,
        og=None
    ):
    """
    Compute the PR-curve metrics for each label, BUT exclude from each
    label's pool of predictions those samples samples for which one of
    the sample's most-specific labels is an ancestor of the current
    label.
    """
    all_labels = set(label_graph.get_all_nodes())
    label_to_assigneds, label_to_scores, item_order = _compute_label_to_item_assignments(
        item_to_label_to_conf,
        item_to_labels,
        restrict_to_labels=restrict_to_labels
    )
    label_to_cons_assigneds = defaultdict(lambda: [])
    label_to_cons_scores = defaultdict(lambda: [])
    global_assigneds = []
    global_scores = []

    blacklist_labels = set([
        frozenset(['CL:0000010']),
        frozenset(['CL:0000578']),
        frozenset(['CL:0001034'])
    ])

    print "Mapping items to their most-specific labels..."
    item_to_ms_labels = {}
    for item, labels in item_to_labels.iteritems():
        labels = label_graph.most_specific_nodes(
            set(labels) & all_labels
        )
        ms_item_labels = label_graph.most_specific_nodes(
            set(labels) & all_labels
        )
        ms_item_labels = ms_item_labels - blacklist_labels
        #if frozenset(["CL:2000001"]) in ms_item_labels:
        #    ms_item_labels.add(
        #        frozenset([u"CL:0000842"]) # mononuclear cell
        #    )
        item_to_ms_labels[item] = ms_item_labels
    print "done."


    # Create new nodes in the label graph corresponding to 
    # joint-labels -- that is, sets of labels for which there
    # exists a sample labelled with both labels. For example,
    # if an experiment is labelled with both 'PBMC' and
    # 'T cell', then we create a new label 'PBMC + T cell'
    mod_label_graph = label_graph.copy()
    joint_labels = set()
    item_to_new_ms_labels = defaultdict(lambda: set())
    mod_label_to_name = {
        label: name
        for label, name in label_to_name.iteritems()
    }
    new_labels = set()
    for item, ms_labels in item_to_ms_labels.iteritems():
        if len(ms_labels) > 1:
            joint_label = frozenset(ms_labels)
            joint_labels.add(joint_label)
            mod_label_to_name[joint_label] = " & ".join([
                mod_label_to_name[ms_label]
                for ms_label in ms_labels
            ])
            item_to_new_ms_labels[item].add(joint_label)
            for ms_label in ms_labels:
                mod_label_graph.add_edge(ms_label, joint_label)

    print "New, joint labels:"
    for joint_label in new_labels:
        "%s with name %s" % (joint_label, mod_label_to_name[joint_label])

    # Make a 'deep' copy of the mappings from experiments to most-specific 
    # labels. Then recompute the most-specifc labels and predictions now 
    # that we have added new labels and predictions
    mod_item_to_ms_labels = {
        item: set(ms_labels)
        for item, ms_labels in item_to_ms_labels.iteritems()
    }
    for item, new_ms_labels in item_to_new_ms_labels.iteritems():
        mod_item_to_ms_labels[item].update(new_ms_labels)
    mod_item_to_ms_labels = {
        item: mod_label_graph.most_specific_nodes(labels)
        for item, labels in mod_item_to_ms_labels.iteritems()
    }

    # If the sample is most-specifically labeled as PBMC, then
    # for our purposes, we treat mononuclear cell as its most
    # specific label 
    item_to_ms_labels = mod_item_to_ms_labels
    for item, ms_labels in item_to_ms_labels.iteritems():
        if frozenset(["CL:2000001"]) in ms_labels:
            ms_labels.add(
                frozenset(["CL:0000842"]) # mononuclear cell
            ) 

    item_to_anc_desc_ms_labels = {}
    for item, ms_labels in item_to_ms_labels.iteritems():
        desc_ms_labels = set()
        for ms_label in ms_labels:
            desc_ms_labels.update(
                mod_label_graph.descendent_nodes(ms_label)
            )
        anc_desc_ms_labels = set()
        for desc_ms_label in desc_ms_labels:
            anc_desc_ms_labels.update(
                mod_label_graph.ancestor_nodes(desc_ms_label)
            )
        anc_desc_ms_labels = anc_desc_ms_labels - set(item_to_labels[item])
        item_to_anc_desc_ms_labels[item] = anc_desc_ms_labels

    skipped_pairs = set()
    pair_to_items = defaultdict(lambda: set())
    for curr_label in all_labels:
        if curr_label not in label_to_assigneds:
            continue
        print "Examining label %s" % pdoo.convert_node_to_name(curr_label, og)
        assigneds = label_to_assigneds[curr_label]
        scores = label_to_scores[curr_label]
        cons_assigneds = []
        cons_scores = []
        skipped = set()
        anc_labels = set(mod_label_graph.ancestor_nodes(curr_label)) - set([curr_label])
        for item, assigned, score in zip(item_order, assigneds, scores):
            #ms_item_labels = label_graph.most_specific_nodes(
            #    set(item_to_labels[item]) & all_labels
            #)
            ## If this is a PBMC sample, then it's most-specific
            ## labels should also include mononuclear cell
            #if frozenset(["CL:2000001"]) in ms_item_labels:
            #    ms_item_labels.add(
            #        frozenset(["CL:0000842"])
            #    )
            ms_labels = item_to_ms_labels[item]
            anc_desc_ms_labels = item_to_anc_desc_ms_labels[item]
            # NOTE this is the crucial step in which we skip
            # samples that have a most-specific label that is
            # an ancestor of the current label
            #if item == "SRX2364422":
            #    print "FOUND ITEM %s" % item
            #    print "Most-specific labels: %s" % [pdoo.convert_node_to_name(x, og) for x in ms_labels]
            #    print ms_labels
            #    print "Ancestor labels: %s" % [pdoo.convert_node_to_name(x, og) for x in anc_labels]
            #    print anc_labels
            #    print "|ms_labels & anc_labels| = %d" % len(set(ms_labels) & anc_labels)
            #    print

            if len(set(ms_labels) & anc_labels) > 0:
                for ms_label in set(ms_labels) & set(anc_labels):
                    skipped_pairs.add((ms_label, curr_label))
                    pair_to_items[(ms_label, curr_label)].add(item)
                skipped.add(item)
                continue
            if curr_label in anc_desc_ms_labels:
                skipped.add(item)
                continue

            cons_assigneds.append(assigned)
            cons_scores.append(score)
            global_assigneds.append(
                curr_label in item_to_labels[item]
            )
            global_scores.append(score)
        label_to_cons_assigneds[curr_label] = cons_assigneds
        label_to_cons_scores[curr_label] = cons_scores
        print "Label %s" % label_to_name[curr_label]
        print "N samples in ranking: %d" % len(cons_assigneds)
        print "N skipped: %d" % len(skipped)
        print "Sample of skipped %s" % list(skipped)[0:20]
        print 
    label_to_assigneds = dict(label_to_cons_assigneds)
    label_to_scores = dict(label_to_cons_scores) 


    print "ms-label\tlabel\tn exps"
    for pair in skipped_pairs:
        print "%s\t%s\t%d" % (pdoo.convert_node_to_name(pair[0], og), pdoo.convert_node_to_name(pair[1], og), len(pair_to_items[pair]))
        #if len(pair_to_items[pair]) < 100:
        #    print pair_to_items[pair]

    label_to_auc, label_to_avg_precision, label_to_pr_curve = _compute_per_label_metrics(
        all_labels,
        label_to_assigneds,
        label_to_scores,
        restrict_to_labels=restrict_to_labels
    )
    global_precisions, global_recalls = _precision_recall_curve(
        global_assigneds, 
        global_scores
    )
    global_pr_curve = (global_precisions, global_recalls)
    global_avg_precision = _average_precision(global_precisions, global_recalls)
    return (
        label_to_auc,
        label_to_avg_precision,
        label_to_pr_curve,
        global_pr_curve,
        global_avg_precision
    )


def compute_per_sample_metrics(
        sample_to_label_to_conf,
        sample_to_labels,
        label_graph,
        sample_to_study,
        study_to_samples,
        label_to_name=None,
        restrict_to_labels=None,
        og=None,
        thresh_range='zero_to_one'
    ):

    # Compute the average per-sample metrics
    #for pred_thresh in np.arange()
    precisions = []
    recalls = []
    sw_precisions = []
    sw_recalls = []
    ms_precisions = []
    ms_recalls = []
    sw_ms_precisions = []
    sw_ms_recalls = []
    if thresh_range == 'zero_to_one':
        pred_threshes = np.concatenate((
            np.arange(0.0, 0.9, 0.05),
            np.arange(0.9, 0.99, 0.01),
            np.arange(0.99, 0.999, 0.001),
            np.arange(0.999, 0.9999, 0.0001),
            np.arange(0.9999, 0.99999, 0.00001),
            np.array([0.99999999]),
            np.array([0.99999999999]),
            np.array([1.0]),
            np.array([1.1]) # To get the PR-point at P=1, R=0
        ))
    elif thresh_range == '1nn':
        pred_threshes = np.concatenate((
            np.array([float('-inf')]),
            np.arange(-2.0, 2.0, 0.01)
        ))
    elif thresh_range == 'data_driven':
        max_conf = None
        min_conf = None
        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            curr_max = max(label_to_conf.values())
            curr_min = min(label_to_conf.values())
            if not max_conf or curr_max > max_conf:
                max_conf = curr_max
            if not min_conf or curr_min < min_conf:
                min_conf = curr_min
        pred_threshes = np.array([min_conf, max_conf, 0.05])
    print "Pred threshes: %s" % pred_threshes

    # Determine which samples have no true labels. Also, precompute
    # the most-specific labels for each sample
    skipped_samples = set()
    sample_to_ms_true_pos_labels = {}
    for sample in sample_to_label_to_conf:
        true_pos_labels = set(sample_to_labels[sample])
        if restrict_to_labels:
            true_pos_labels &= restrict_to_labels
        if len(true_pos_labels) == 0:
            skipped_samples.add(sample)
        ms_true_pos_labels = label_graph.most_specific_nodes(true_pos_labels)
        if restrict_to_labels:
            ms_true_pos_labels &= restrict_to_labels
        sample_to_ms_true_pos_labels[sample] = ms_true_pos_labels
    print "%d samples have no true labels." % len(skipped_samples)

    for pred_thresh in pred_threshes:
        all_studies = set()
        n_samples = 0
        avg_sample_prec_sum = 0.0
        avg_sample_recall_sum = 0.0
        avg_sw_sample_prec_sum = 0.0
        avg_sw_sample_recall_sum = 0.0
        avg_sample_ms_prec_sum = 0.0
        avg_sample_ms_recall_sum = 0.0
        avg_sw_sample_ms_prec_sum = 0.0
        avg_sw_sample_ms_recall_sum = 0.0

        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            if sample in skipped_samples:
                continue
            n_samples += 1.0

            pred_pos_labels = set([
                label
                for label, conf in label_to_conf.iteritems()
                if conf >= pred_thresh
            ])
            ms_pred_pos_labels = label_graph.most_specific_nodes(pred_pos_labels)
            true_pos_labels = set(sample_to_labels[sample])
            ms_true_pos_labels = sample_to_ms_true_pos_labels[sample]
            if restrict_to_labels:
                true_pos_labels &= restrict_to_labels
                pred_pos_labels &= restrict_to_labels
                ms_pred_pos_labels &= restrict_to_labels
            study = sample_to_study[sample]
            study_weight = 1.0 / len(study_to_samples[study] - skipped_samples)
            all_studies.add(study)
            if len(pred_pos_labels) > 0:
                sample_prec = len(pred_pos_labels & true_pos_labels) / float(len(pred_pos_labels))
            else:
                sample_prec = 1.0
            sw_sample_prec = sample_prec * study_weight
            avg_sample_prec_sum += sample_prec
            avg_sw_sample_prec_sum += sw_sample_prec

            if len(true_pos_labels) > 0:
                sample_recall = len(pred_pos_labels & true_pos_labels) / float(len(true_pos_labels))
            else:
                sample_recall = 1.0
            sw_sample_recall = sample_recall * study_weight
            avg_sample_recall_sum += sample_recall
            avg_sw_sample_recall_sum += sw_sample_recall

            if len(ms_pred_pos_labels) > 0:
                sample_ms_prec = len(ms_pred_pos_labels & true_pos_labels) / float(len(ms_pred_pos_labels))
            else:
                sample_ms_prec = 1.0
            sw_sample_ms_prec = sample_ms_prec * study_weight 
            avg_sample_ms_prec_sum += sample_ms_prec
            avg_sw_sample_ms_prec_sum += sw_sample_ms_prec

            if len(ms_true_pos_labels) > 0:
                sample_ms_recall = len(pred_pos_labels & ms_true_pos_labels) / float(len(ms_true_pos_labels))
            else:
                sample_ms_recall = 1.0
            sw_sample_ms_recall = sample_ms_recall * study_weight
            avg_sample_ms_recall_sum += sample_ms_recall
            avg_sw_sample_ms_recall_sum += sw_sample_ms_recall


            #if pred_thresh == 0.9:
            #    print "Sample: %s" % sample
            #    print "True labels: %s" % [label_to_name[x] for x in true_pos_labels]
            #    print "Predicted labels: %s" % [label_to_name[x] for x in pred_pos_labels]
            #    print "Most-specific true labels: %s" % [label_to_name[x] for x in ms_true_pos_labels]
            #    print "Most-specific predicted labels: %s" % [label_to_name[x] for x in ms_pred_pos_labels] 
            #    print "Precision: %f" % sample_prec
            #    print "Recall: %f" % sample_recall
            #    print "Specific terms precision: %f" % sample_ms_prec
            #    print "Specific terms recall: %f" % sample_ms_recall
            #    print

        n_studies = len(all_studies)
        avg_sample_prec = avg_sample_prec_sum / n_samples 
        avg_sample_recall = avg_sample_recall_sum  / n_samples
        avg_sw_sample_prec = avg_sw_sample_prec_sum / n_studies
        avg_sw_sample_recall = avg_sw_sample_recall_sum / n_studies
        avg_sample_ms_prec = avg_sample_ms_prec_sum / n_samples
        avg_sample_ms_recall = avg_sample_ms_recall_sum / n_samples
        avg_sw_sample_ms_prec = avg_sw_sample_ms_prec_sum / n_studies
        avg_sw_sample_ms_recall = avg_sw_sample_ms_recall_sum / n_studies
       
        print "Threshold is %f. Prec: %f. Recall: %f. Ms-prec: %f. Ms-recall: %f" % (
            pred_thresh,
            avg_sample_prec,
            avg_sample_recall,
            avg_sample_ms_prec,
            avg_sample_ms_recall
        )
        precisions.append(avg_sample_prec)
        recalls.append(avg_sample_recall)
        sw_precisions.append(avg_sw_sample_prec)
        sw_recalls.append(avg_sw_sample_recall)
        ms_precisions.append(avg_sample_ms_prec)
        ms_recalls.append(avg_sample_ms_recall)
        sw_ms_precisions.append(avg_sw_sample_ms_prec)
        sw_ms_recalls.append(avg_sw_sample_ms_recall)
    return (
        None,   # TODO REMOVE THIS
        None,   # TODO REMOVE THIS
        (precisions, recalls),
        (sw_precisions, sw_recalls),
        (ms_precisions, ms_recalls),
        (sw_ms_precisions, sw_ms_recalls)
    )
 

def compute_conservative_per_sample_metrics(
        sample_to_label_to_conf,
        sample_to_labels,
        label_graph,
        sample_to_study,
        study_to_samples,
        label_to_name=None,
        restrict_to_labels=None,
        og=None,
        thresh_range='zero_to_one'
    ):

    # Compute the average per-sample metrics
    #for pred_thresh in np.arange()
    precisions = []
    recalls = []
    sw_precisions = []
    sw_recalls = []
    ms_precisions = []
    ms_recalls = []
    sw_ms_precisions = []
    sw_ms_recalls = []
    if thresh_range == 'zero_to_one':
        pred_threshes = np.concatenate((
            np.arange(0.0, 0.9, 0.05),
            np.arange(0.9, 0.99, 0.01),
            np.arange(0.99, 0.999, 0.001),
            np.arange(0.999, 0.9999, 0.0001),
            np.arange(0.9999, 0.99999, 0.00001),
            np.array([0.99999999]),
            np.array([0.99999999999]),
            np.array([1.0]),
            np.array([1.1]) # To get the PR-point at P=1, R=0
        ))
    elif thresh_range == '1nn':
        pred_threshes = np.concatenate((
            np.array([float('-inf')]),
            np.arange(-2.0, 2.0, 0.01)
        ))
    elif thresh_range == 'data_driven':
        max_conf = None
        min_conf = None
        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            curr_max = max(label_to_conf.values())
            curr_min = min(label_to_conf.values())
            if not max_conf or curr_max > max_conf:
                max_conf = curr_max
            if not min_conf or curr_min < min_conf:
                min_conf = curr_min
        pred_threshes = np.array([min_conf, max_conf, 0.05])
    print "Pred threshes: %s" % pred_threshes

    # Determine which samples have no true labels. Also, precompute
    # the most-specific labels for each sample
    skipped_samples = set()
    sample_to_ms_true_pos_labels = {}
    sample_to_ignore_labels = {}
    for sample in sample_to_label_to_conf:
        true_pos_labels = set(sample_to_labels[sample])
        if restrict_to_labels:
            true_pos_labels &= restrict_to_labels
        if len(true_pos_labels) == 0:
            skipped_samples.add(sample)
        ms_true_pos_labels = label_graph.most_specific_nodes(true_pos_labels)
        if restrict_to_labels:
            ms_true_pos_labels &= restrict_to_labels
        sample_to_ms_true_pos_labels[sample] = ms_true_pos_labels

        # Precompute the labels that are "ambiguous". That is, labels that are
        # are more-specific than the most-specific for the sample and ancestors
        # of these terms (that aren't in the true labels set).
        desc_ms_labels = set()
        for ms_label in ms_true_pos_labels:
            desc_ms_labels.update(
                label_graph.descendent_nodes(ms_label)
            )
        anc_desc_ms_labels = set()
        for desc_ms_label in desc_ms_labels:
            anc_desc_ms_labels.update(
                label_graph.ancestor_nodes(desc_ms_label)
            )
        ignore_labels = anc_desc_ms_labels - true_pos_labels
        sample_to_ignore_labels[sample] = ignore_labels    

    print "%d samples have no true labels." % len(skipped_samples)

    # REMOVEEEEEEEEEEEE
    #samples_w_ignore = [
    #    sample
    #    for sample, ignore in sample_to_ignore_labels.iteritems() 
    #    if len(ignore) > 2
    #]
    #print "Samples and ignore labels"
    #for sample in samples_w_ignore[:10]:
    #    print "%s\t%s" % (sample, [label_to_name[x] for x in sample_to_ignore_labels[sample]])
    #print "%d samples have labels to ignore" % len(samples_w_ignore)
    # REMOVEEEEE

    for pred_i, pred_thresh in enumerate(pred_threshes):
        all_studies = set()
        n_samples = 0
        avg_sample_prec_sum = 0.0
        avg_sample_recall_sum = 0.0
        avg_sw_sample_prec_sum = 0.0
        avg_sw_sample_recall_sum = 0.0
        avg_sample_ms_prec_sum = 0.0
        avg_sample_ms_recall_sum = 0.0
        avg_sw_sample_ms_prec_sum = 0.0
        avg_sw_sample_ms_recall_sum = 0.0

        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            if sample in skipped_samples:
                continue
            n_samples += 1.0

            pred_pos_labels = set([
                label
                for label, conf in label_to_conf.iteritems()
                if conf >= pred_thresh
            ])
            ignore_labels = sample_to_ignore_labels[sample]
            pred_pos_labels -= ignore_labels
            ms_pred_pos_labels = label_graph.most_specific_nodes(pred_pos_labels)
            true_pos_labels = set(sample_to_labels[sample])
            ms_true_pos_labels = sample_to_ms_true_pos_labels[sample]
            if restrict_to_labels:
                true_pos_labels &= restrict_to_labels
                pred_pos_labels &= restrict_to_labels
                ms_pred_pos_labels &= restrict_to_labels
            #pred_pos_labels -= ignore_labels
            #ms_pred_pos_labels -= ignore_labels
            
            study = sample_to_study[sample]
            study_weight = 1.0 / len(study_to_samples[study] - skipped_samples)
            all_studies.add(study)
            if len(pred_pos_labels) > 0:
                sample_prec = len(pred_pos_labels & true_pos_labels) / float(len(pred_pos_labels))
            else:
                sample_prec = 1.0
            sw_sample_prec = sample_prec * study_weight
            avg_sample_prec_sum += sample_prec
            avg_sw_sample_prec_sum += sw_sample_prec

            if len(true_pos_labels) > 0:
                sample_recall = len(pred_pos_labels & true_pos_labels) / float(len(true_pos_labels))
            else:
                sample_recall = 1.0
            sw_sample_recall = sample_recall * study_weight
            avg_sample_recall_sum += sample_recall
            avg_sw_sample_recall_sum += sw_sample_recall

            if len(ms_pred_pos_labels) > 0:
                sample_ms_prec = len(ms_pred_pos_labels & true_pos_labels) / float(len(ms_pred_pos_labels))
            else:
                sample_ms_prec = 1.0
            sw_sample_ms_prec = sample_ms_prec * study_weight
            avg_sample_ms_prec_sum += sample_ms_prec
            avg_sw_sample_ms_prec_sum += sw_sample_ms_prec

            if len(ms_true_pos_labels) > 0:
                sample_ms_recall = len(pred_pos_labels & ms_true_pos_labels) / float(len(ms_true_pos_labels))
            else:
                sample_ms_recall = 1.0
            sw_sample_ms_recall = sample_ms_recall * study_weight
            avg_sample_ms_recall_sum += sample_ms_recall
            avg_sw_sample_ms_recall_sum += sw_sample_ms_recall


            #if pred_thresh == 0.2:
            #    print "Sample: %s" % sample
            #    print "True labels: %s" % [label_to_name[x] for x in true_pos_labels]
            #    print "Predicted labels: %s" % [label_to_name[x] for x in pred_pos_labels]
            #    print "Most-specific true labels: %s" % [label_to_name[x] for x in ms_true_pos_labels]
            #    print "Most-specific predicted labels: %s" % [label_to_name[x] for x in ms_pred_pos_labels] 
            #    print "Precision: %f" % sample_prec
            #    print "Recall: %f" % sample_recall
            #    print "Specific terms precision: %f" % sample_ms_prec
            #    print "Specific terms recall: %f" % sample_ms_recall
            #    print

        n_studies = len(all_studies)
        avg_sample_prec = avg_sample_prec_sum / n_samples
        avg_sample_recall = avg_sample_recall_sum  / n_samples
        avg_sw_sample_prec = avg_sw_sample_prec_sum / n_studies
        avg_sw_sample_recall = avg_sw_sample_recall_sum / n_studies
        avg_sample_ms_prec = avg_sample_ms_prec_sum / n_samples
        avg_sample_ms_recall = avg_sample_ms_recall_sum / n_samples
        avg_sw_sample_ms_prec = avg_sw_sample_ms_prec_sum / n_studies
        avg_sw_sample_ms_recall = avg_sw_sample_ms_recall_sum / n_studies

        print "Threshold is %f. Prec: %f. Recall: %f. Ms-prec: %f. Ms-recall: %f" % (
            pred_thresh,
            avg_sample_prec,
            avg_sample_recall,
            avg_sample_ms_prec,
            avg_sample_ms_recall
        )
        precisions.append(avg_sample_prec)
        recalls.append(avg_sample_recall)
        sw_precisions.append(avg_sw_sample_prec)
        sw_recalls.append(avg_sw_sample_recall)
        ms_precisions.append(avg_sample_ms_prec)
        ms_recalls.append(avg_sample_ms_recall)
        sw_ms_precisions.append(avg_sw_sample_ms_prec)
        sw_ms_recalls.append(avg_sw_sample_ms_recall)
    return (
        None,   # TODO REMOVE THIS
        None,   # TODO REMOVE THIS
        (precisions, recalls),
        (sw_precisions, sw_recalls),
        (ms_precisions, ms_recalls),
        (sw_ms_precisions, sw_ms_recalls)
    )

def _compute_cluster_weighted_metric(
        cluster_to_labels, 
        clusters, 
        label_to_metric
    ):
    clust_metrics = []
    for clust in clusters:
        clust_sum = float(sum([
            label_to_metric[label]
            for label in cluster_to_labels[clust]
            if label in label_to_metric
        ]))
        clust_metric = clust_sum / len(cluster_to_labels[clust])
        clust_metrics.append(clust_metric)
    return sum(clust_metrics) / len(clust_metrics)
    

def dendrogram_metrics(
        label_to_metric,
        cluster_to_height,
        cluster_to_labels
    ):
    clust_w_heights = sorted(
        [
            (clust, height)
            for clust, height in cluster_to_height.iteritems()
        ],
        key=lambda x: x[1]
    )
    curr_height = clust_w_heights[0][1]
    heights = []
    metrics = []
    for i, clust_w_height in enumerate(clust_w_heights):
        height = clust_w_height[1]
        if height == curr_height:
            continue
        clusts = [
            x[0]
            for x in clust_w_heights[0:i+1]
        ]
        weighted_metric = _compute_cluster_weighted_metric(
            cluster_to_labels,
            clusts,
            label_to_metric
        )
        heights.append(height)
        metrics.append(weighted_metric)
        curr_height = height
    print "HEIGHTS: %s" % heights
    print "METRICS: %s" % metrics
    return heights, metrics
  
 


def _precision(preds_w_trues):
    true_pos = float(len([
        x
        for x in preds_w_trues
        if x[1] and x[0]              
    ]))
    pred_pos = float(len([
        x
        for x in preds_w_trues
        if x[0]
    ]))
    if pred_pos == 0:
        return 0.0
    else:
        return true_pos / pred_pos
    

def _recall(preds_w_trues):
    true_pos = float(len([
        x
        for x in preds_w_trues
        if x[1] and x[0]              
    ]))
    all_pos = float(len([
        x
        for x in preds_w_trues
        if x[1]
    ]))
    return true_pos / all_pos
    

def _precision_recall_curve_OLD(classes, scores):
    scores_w_classes = zip(scores, classes)
    scores_w_classes = sorted(scores_w_classes, key=lambda x: x[0])
    curr_score = None
    precisions = []
    recalls = []
    for i, score_w_class in enumerate(scores_w_classes):
        if curr_score != score_w_class[0] and score_w_class[0] != float('-inf'):
            preds_w_trues = [
                (False, swc[1])
                for swc in scores_w_classes[0:i]
            ]
            preds_w_trues += [
                (True, swc[1])
                for swc in scores_w_classes[i:]
            ]
            prec = _precision(preds_w_trues)
            recall = _recall(preds_w_trues)
            precisions.append(prec)
            recalls.append(recall)
            curr_score = score_w_class[0]
    precisions.reverse()
    recalls.reverse()
    return precisions, recalls


def _precision_recall_curve(classes, scores):
    scores_w_classes = zip(scores, classes)
    scores_w_classes = sorted(scores_w_classes, key=lambda x: x[0])
    curr_score = None
    precisions = []
    recalls = []


    total_pos = float(len([x for x in scores_w_classes if x[1]]))
    pos_seen = 0.0
    last_score = None
    for i, score_w_class in enumerate(scores_w_classes[:-1]):
        curr_score = score_w_class[0]
        if score_w_class[1]: # Positive instance
            pos_seen += 1.0
        if last_score is None or curr_score != last_score:
            prec = (total_pos  - pos_seen) / float(len(scores_w_classes) - (i+1))
            recall = (total_pos  - pos_seen) / total_pos
            precisions.append(prec)
            recalls.append(recall)
        last_score = curr_score
    precisions.reverse()
    recalls.reverse()
    return precisions, recalls


def _average_precision(precisions, recalls):
    summ = 0.0
    #print "PRECISIONS: %s" % precisions
    #print "RECALLS: %s" % recalls
    if len(recalls) > 0 and recalls[0] != 0.0:
        precisions = [0.0] + precisions
        recalls = [0.0] + recalls
    #print "Computing average precision. The recalls are:"
    #print recalls
    #print "The precisions are:"
    #print precisions
    for i in range(1, len(recalls)):
        recall_prev = recalls[i-1]
        recall = recalls[i]
        prec = precisions[i]
        summ += (recall - recall_prev)*prec
    #    print "Summ=%f += (recall=%f - recall_prev=%f)*prec=%f" % (summ, recall, recall_prev, prec)
    return summ


def label_hiearchical_clustering(sample_to_labels):
    label_to_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_samples[label].add(sample)
    label_to_samples = dict(label_to_samples)

    clust_id = 0

    # Maps each cluster to the set of labels it includes
    clust_to_labels = {}
    for label in label_to_samples:
        clust_to_labels[clust_id] = frozenset([label])
        clust_id += 1
    clust_to_clust_to_dist = defaultdict(lambda: {})
    clust_to_children = {k:frozenset() for k in clust_to_labels}   

    # Initialize the distance matrix
    for clust_1, labels_1 in clust_to_labels.iteritems():
        for clust_2, labels_2 in clust_to_labels.iteritems():
            samples_1 = set(label_to_samples[list(labels_1)[0]])
            samples_2 = set(label_to_samples[list(labels_2)[0]])
            if clust_1 == clust_2:
                continue
            # Jaccard distance
            dist = 1.0 - (len(samples_1 & samples_2) / float(len(samples_1 | samples_2)))
            print "dist between clust %s and %s is %f" % (clust_1, clust_2, dist)
            clust_to_clust_to_dist[clust_1][clust_2] = dist
            clust_to_clust_to_dist[clust_2][clust_1] = dist
    clust_to_height = {k:0.0 for k in clust_to_labels}

    joinable_clusts = set(clust_to_labels.keys()) 
    while len(joinable_clusts) > 1:
        # Find the two meta-labels with minimal distance
        min_dist = float('inf')
        clust_1_min = None
        clust_2_min = None
        for clust_1 in joinable_clusts:
            for clust_2 in joinable_clusts:
                if clust_1 == clust_2:
                    continue
                if clust_to_clust_to_dist[clust_1][clust_2] < min_dist:
                    min_dist = clust_to_clust_to_dist[clust_1][clust_2]
                    clust_1_min = clust_1
                    clust_2_min = clust_2
         
        # Merge the closest clusters
        new_clust = clust_id
        clust_id += 1
        print "Merging clusters %s and %s" % (clust_1_min, clust_2_min)
        clust_to_height[new_clust] = clust_to_height[clust_1_min] + min_dist
        clust_to_labels[new_clust] = clust_to_labels[clust_1_min] | clust_to_labels[clust_2_min]
        clust_to_children[new_clust] = frozenset([clust_1_min, clust_2_min])
        joinable_clusts.add(new_clust)
        update_clusts = joinable_clusts - set([clust_1_min, clust_2_min, new_clust]) 
        for clust in update_clusts:
            # the UPGMA formula
            dist = (len(clust_to_labels[clust_1_min]) * clust_to_clust_to_dist[clust][clust_1_min] \
                + len(clust_to_labels[clust_2_min]) * clust_to_clust_to_dist[clust][clust_2_min]) \
                / float(len(clust_to_labels[clust_1_min]) + len(clust_to_labels[clust_2_min]))
            print "Computed distance %f" % dist
            clust_to_clust_to_dist[clust][new_clust] = dist
            clust_to_clust_to_dist[new_clust][clust] = dist

        # Delete the old clusters from the distance matrix
        # and from set of clusters that are eligible for joining
        joinable_clusts.remove(clust_1_min)
        joinable_clusts.remove(clust_2_min)
        for clust in update_clusts:
            clust_to_dist = clust_to_clust_to_dist[clust]
            if clust != clust_1_min and clust != new_clust:
                del clust_to_dist[clust_1_min]
            if clust != clust_2_min and clust != new_clust:
                del clust_to_dist[clust_2_min]
        del clust_to_clust_to_dist[clust_1_min]
        del clust_to_clust_to_dist[clust_2_min]
    
    return clust_to_children, clust_to_height, clust_to_labels 
        
    
def compute_included_labels(
        the_exps,
        exp_to_labels,
        exp_to_info,
        og
    ):
    label_to_num_studies = data_metrics(
        the_exps,
        exp_to_labels,
        exp_to_info
    )
    include_labels = set([
        label
        for label, n_studies in label_to_num_studies.iteritems()
        if n_studies > 1
    ])

    print "Label to number of studies:"
    print label_to_num_studies

    # Compute the trivial labels
    trivial_labels = set(
        exp_to_labels[
            list(exp_to_labels.keys())[0]
        ]
    )
    for labels in exp_to_labels.values():
        trivial_labels = trivial_labels & set(labels)
    print "Computed the following trivial labels: %s" % trivial_labels

    include_labels -= trivial_labels
    return include_labels

















def convert_keys_to_label_name(label_to_value, label_to_name):
    return {
        label_to_name[k]: v
        for k,v in label_to_value.iteritems()
    }


def validate_predictions(
        sample_to_label_to_confidence, 
        include_labels
    ):    
    """
    Ensure that each sample has predictions for 
    the set of labels we want to compute metrics
    for
    """
    samples_w_preds = list(sample_to_label_to_confidence.keys())
    for sample, label_to_confs in sample_to_label_to_confidence.iteritems():
        curr_labels = frozenset(label_to_confs.keys())
        if not curr_labels >= include_labels:
            print "Oh no!!! I don't have the following labels in experiment %s:" % sample
            print (include_labels - curr_labels)
        assert curr_labels >= include_labels












#def evaluate_hard_predictions(): # TODO Write the method signature
#
#    sample_to_predictsion = apply_prediction_threshold(
#        sample_to_label_to_confidence, 
#        thresh
#    )
#
#
#    # convert predictions
#    sample_to_predictions = {
#        sample: set([
#            pdoo.convert_node_from_str(x)
#            for x in pred  
#        ])
#        for sample, pred in sample_to_predictions.iteritems()
#    }
#
#    # assure prediction samples are same as those labelled
#    sample_to_labels = {k:v for k,v in sample_to_labels.iteritems() if k in sample_to_predictions}
#    assert frozenset(sample_to_predictions.keys()) == frozenset(sample_to_labels.keys())
#
#    print "TOTAL SAMPLES: %s" % len(sample_to_predictions)
#
#    sample_to_info = data_info['sample_info']
#    sample_accs_w_data = data_info['samples_with_recount_data']
#
#    # Map study to samples in the cross-validation set on 
#    # which predictions were made on
#    study_to_samples = defaultdict(lambda: set())
#    included_studies = set()
#    for data_part in data_partitions:
#        included_studies.update(data_partitions[data_part])
#    for sample, sample_dat in sample_to_info.iteritems():
#        study = sample_dat['study_accession']
#        if sample in sample_to_predictions and study in included_studies:
#            study_to_samples[study].add(sample)
#
#    sample_accs = set()
#    for study, samples in study_to_samples.iteritems():
#        sample_accs.update(samples)
#
#    label_to_studies = defaultdict(lambda: set())
#    for sample, labels in sample_to_labels.iteritems():
#        study = sample_to_info[sample]['study_accession']
#        for label in labels:
#            label_to_studies[label].add(study)
#        
#    sample_to_false_positives, sample_to_false_negatives, sample_to_hamming_loss = compute_per_sample_performance(
#        sample_to_predictions,
#        sample_to_labels
#    )
#
#    label_to_per_sample_recall = per_label_recall(
#        sample_to_predictions,
#        sample_to_labels
#    )
#
#    print "LABEL TO PER SAMPLE RECALL"
#    print label_to_per_sample_recall
#
#    label_to_per_sample_prec = per_label_precision(
#        sample_to_predictions,
#        sample_to_labels
#    )
#
#    # Compute the per-sample 'weight'. A sample's weight is
#    # 1/num_samples where num_samples is number of samples
#    # in that sample's study    
#    sample_to_weight_recall = {}
#    for study, samples in study_to_samples.iteritems():
#        for sample in samples:
#            sample_to_weight_recall[sample] = 1.0/len(samples)
#
#
#
#    # TODO MAKE SURE THIS IS WORKING  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    # Compute the per-sample 'weight' for each study to be 
#    # used in the calculation of precision for the first 
#    # method.
#    label_to_sample_to_weight_prec = defaultdict(lambda: {})
#    pred_label_to_studies = defaultdict(lambda: set())
#    for sample, preds in sample_to_predictions.iteritems():
#        study = sample_to_info[sample]['study_accession']
#        for label in preds:
#            pred_label_to_studies[label].add(study)
#    pred_label_to_samples = defaultdict(lambda: set())
#    for sample, preds in sample_to_predictions.iteritems():
#        for label in preds:
#            pred_label_to_samples[label].add(sample)
#    for label in pred_label_to_samples:
#        for study in pred_label_to_studies[label]:
#            samples_of_label_in_study = set(pred_label_to_samples[label]) & set(study_to_samples[study])
#            for sample in samples_of_label_in_study:
#                label_to_sample_to_weight_prec[label][sample] = 1.0/len(samples_of_label_in_study)
3    # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#    label_to_per_study_recall = per_label_recall(
#        sample_to_predictions,
#        sample_to_labels,
#        sample_to_weight_recall
#    )
#
#    label_to_per_study_prec = per_label_precision(
#        sample_to_predictions,
#        sample_to_labels,
#        label_to_sample_to_weight_prec
#    )
#
#    # Map each label to set of relavents samples and studies
#    label_to_samples = defaultdict(lambda: set())
#    label_to_studies = defaultdict(lambda: set())
#    for sample, labels in sample_to_labels.iteritems():
#        study = sample_to_info[sample]['study_accession']
#        for label in labels:
#            label_to_samples[label].add(sample)
#            label_to_studies[label].add(study)
#
##    plot_study_to_avg_hamming_loss(
##        study_to_avg_hamming_loss,
##        study_to_samples,
##        out_dir
##    )
#
#    # We only want to plot labels that are represented by
#    # at least two studies in the cross-validation set. These
#    # will be labels that at least have the chance of being 
#    # called correctly
#    plot_labels = set([
#        x for x in label_to_studies
#        if len(label_to_studies[x]) >= 2
#    ])
#   
# 
#    plot_label_to_precision_recall_per_sample_per_study(
#        label_to_per_sample_recall,
#        label_to_per_study_recall,
#        label_to_per_sample_prec,
#        label_to_per_study_prec,
#        label_to_samples,
#        label_to_studies,
#        out_dir,
#        plot_labels=plot_labels
#    )
#
#    """    
#    plot_label_to_precision_recall_per_sample(
#        label_to_per_sample_recall,
#        label_to_per_study_recall,
#        label_to_per_sample_prec,
#        label_to_per_study_prec,
#        label_to_samples,
#        label_to_studies,
#        out_dir,
#        plot_labels=plot_labels
#    )
#    """
#
#    """
#    plot_confusion_like_matrix( 
#        sample_to_labels,
#        sample_to_predictions,
#        out_dir,
#        plot_labels=plot_labels
#    )
#    """
#
#    print "Total number of studies: %d" % len(study_to_samples)
#    print "Total number of samples: %d" % len(sample_accs)
#    print "Total number of labels: %d" % len(label_to_samples)


def compute_per_sample_performance(
        sample_to_predictions,
        sample_to_labels
    ):
    sample_to_false_positives = defaultdict(lambda: set())
    sample_to_false_negatives = defaultdict(lambda: set())
    sample_to_hamming_loss = {}
    for sample, predictions in sample_to_predictions.iteritems():
        predictions = set(predictions)
        labels = set(sample_to_labels[sample])
        false_positives = predictions - labels
        false_negatives = labels - predictions
        true_positives = labels & predictions
        hamming_loss = len(false_positives) + len(false_negatives)

        sample_to_false_positives[sample] = false_positives
        sample_to_false_negatives[sample] = false_negatives
        sample_to_hamming_loss[sample] = hamming_loss
    return sample_to_false_positives, sample_to_false_negatives, sample_to_hamming_loss


def per_label_recall(
        sample_to_predictions,
        sample_to_labels,
        sample_to_weight=None
    ):

    if not sample_to_weight:
        sample_to_weight = defaultdict(lambda: 1)

    label_to_predicted_samples = defaultdict(lambda: set())
    for sample, predictions in sample_to_predictions.iteritems():
        for label in predictions:
            label_to_predicted_samples[label].add(sample)

    label_to_relavent_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_relavent_samples[label].add(sample)

    label_to_recall = {}
    for label in label_to_relavent_samples:
        rel_samples = label_to_relavent_samples[label]
        pred_samples = label_to_predicted_samples[label]
        true_positives = rel_samples & pred_samples
        num = float(sum([
            sample_to_weight[x]
            for x in true_positives
        ]))
        den = float(sum([
            sample_to_weight[x]
            for x in rel_samples
        ]))
        recall = num/den
        label_to_recall[label] = recall

    return label_to_recall


def per_label_precision(
        sample_to_predictions,
        sample_to_labels,
        label_to_sample_to_weight=None
    ):

    if not label_to_sample_to_weight:
        label_to_sample_to_weight = defaultdict(lambda: defaultdict(lambda: 1))

    label_to_predicted_samples = defaultdict(lambda: set())
    for sample, predictions in sample_to_predictions.iteritems():
        for label in predictions:
            label_to_predicted_samples[label].add(sample)

    label_to_relavent_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_relavent_samples[label].add(sample)

    label_to_precision = {}
    for label in label_to_relavent_samples:
        rel_samples = label_to_relavent_samples[label]
        pred_samples = label_to_predicted_samples[label]
        true_positives = rel_samples & pred_samples
        num = float(sum([
            label_to_sample_to_weight[label][x]
            for x in true_positives
        ]))
        den = float(sum([
            label_to_sample_to_weight[label][x]
            for x in pred_samples
        ]))
        if den == 0:
            precision = 0.0
        else:
            precision = num/den
        label_to_precision[label] = precision

    return label_to_precision





########### Functions for making plots

""# TODO THIS WAS ONLY FOR THE PROJECT PROPOSAL
def plot_label_to_precision_recall_per_sample(
        label_to_per_sample_recall,
        label_to_per_study_recall,
        label_to_per_sample_prec,
        label_to_per_study_prec,
        label_to_samples,
        label_to_studies,
        out_dir,
        plot_labels=None
    ):

    if not plot_labels:
        plot_labels = set(label_to_samples.keys()) # Use all of the labels


    random_labels = set(
        random.sample(
            label_to_per_sample_recall.keys(), 
            len(label_to_per_sample_recall)/4
        )
    )

    da_label_per_sample_recall = [
        (ONT_ID_TO_OG["17"].id_to_term[label].name, recall)
        for label,recall in label_to_per_sample_recall.iteritems()
        if label in plot_labels
        and label in random_labels
    ]
    da_label_per_sample_recall = sorted(
        da_label_per_sample_recall,
        key=lambda x: x[1],
        reverse=True
    )
    df_label_per_sample_recall = pandas.DataFrame(
        data=da_label_per_sample_recall,
        columns=["Cell-type", "Recall"]
    )

    label_to_index = {
        x[0]:i
        for i,x in enumerate(da_label_per_sample_recall)
    }

    da_label_per_sample_prec = [
        (ONT_ID_TO_OG["17"].id_to_term[label].name, prec)
        for label,prec in label_to_per_sample_prec.iteritems()
        if label in plot_labels
        and label in random_labels
    ]
    da_label_per_sample_prec = sorted(
        da_label_per_sample_prec,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_per_sample_prec = pandas.DataFrame(
        data=da_label_per_sample_prec,
        columns=["Cell-type", "Precision"]
    )

    da_label_num_samples =  [
        (ONT_ID_TO_OG["17"].id_to_term[label].name, len(num_samples))
        for label, num_samples in label_to_samples.iteritems()
        if label in plot_labels
        and label in random_labels
    ]
    da_label_num_samples = sorted(
        da_label_num_samples,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_num_samples = pandas.DataFrame(
        data=da_label_num_samples,
        columns=["Cell-type", "Number of samples"]
    )

    sns.set(style="whitegrid")
    ax = sns.plt.axes()
    fig, axarr = plt.subplots(
        1,
        3,
        sharey=True,
        figsize=(18, 1.0 + (0.2 * len(da_label_per_sample_recall)))
    )

    vis_lib.single_var_strip_plot(
        df_label_per_sample_recall,
        "Recall",
        "Cell-type",
        x_lims=(0,1),
        ax=axarr[0]
    )

    vis_lib.single_var_strip_plot(
        df_label_per_sample_prec,
        "Precision",
        "Cell-type",
        x_lims=(0,1),
        ax=axarr[1]
    )

    vis_lib.single_var_strip_plot(
        df_label_num_samples,
        "Number of samples",
        "Cell-type",
        x_lims=(0,3000),
        x_ticks=range(0,3500,500),
        ax=axarr[2]
    )

    out_f = join(out_dir, "label_to_precision_recall_per_sample.pdf")
    fig.savefig(out_f, format='pdf', bbox_inches='tight', dpi=200)



def plot_label_to_precision_recall_per_sample_per_study(
        label_to_per_sample_recall,
        label_to_per_study_recall,
        label_to_per_sample_prec,
        label_to_per_study_prec,
        label_to_samples,
        label_to_studies,
        out_dir,
        plot_labels=None
    ):

    def label_to_name(label):
        print "THE LABEL IS %s" % label
        ms_terms = ontology_graph.most_specific_terms(label, ONT_ID_TO_OG["17"])
        ms_term = sorted(ms_terms)[0]
        name = ONT_ID_TO_OG["17"].id_to_term[ms_term].name
        return "%s (%d)" % (name, len(label))

    if not plot_labels:
        plot_labels = set(label_to_samples.keys()) # Use all of the labels
    
    da_label_per_sample_recall = [
        (label_to_name(label), recall)
        for label,recall in label_to_per_sample_recall.iteritems()
        if label in plot_labels
    ]
    da_label_per_sample_recall = sorted(
        da_label_per_sample_recall, 
        key=lambda x: x[1], 
        reverse=True
    )
    df_label_per_sample_recall = pandas.DataFrame(
        data=da_label_per_sample_recall,
        columns=["Label", "Per-sample recall"]
    )

    label_to_index = {
        x[0]:i
        for i,x in enumerate(da_label_per_sample_recall)
    }
    da_label_per_study_recall = [
        (label_to_name(label), recall)
        for label,recall in label_to_per_study_recall.iteritems()
        if label in plot_labels
    ]
    da_label_per_study_recall = sorted(
        da_label_per_study_recall,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_per_study_recall = pandas.DataFrame(
        data=da_label_per_study_recall,
        columns=["Label", "Per-study recall"]
    )

    da_label_per_sample_prec = [
        (label_to_name(label), prec)
        for label,prec in label_to_per_sample_prec.iteritems()
        if label in plot_labels
    ]
    da_label_per_sample_prec = sorted(
        da_label_per_sample_prec,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_per_sample_prec = pandas.DataFrame(
        data=da_label_per_sample_prec,
        columns=["Label", "Per-sample precision"]
    )


    da_label_per_study_prec = [
        (label_to_name(label), prec)
        for label,prec in label_to_per_study_prec.iteritems()
        if label in plot_labels
    ]
    da_label_per_study_prec = sorted(
        da_label_per_study_prec,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_per_study_prec = pandas.DataFrame(
        data=da_label_per_study_prec,
        columns=["Label", "Per-study precision"]
    )


    da_label_num_samples =  [
        (label_to_name(label), len(num_samples))
        for label, num_samples in label_to_samples.iteritems()
        if label in plot_labels
    ]
    da_label_num_samples = sorted(
        da_label_num_samples,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_num_samples = pandas.DataFrame(
        data=da_label_num_samples,
        columns=["Label", "Number of samples"]
    )

    da_label_num_studies =  [
        (label_to_name(label), len(num_studies))
        for label, num_studies in label_to_studies.iteritems()
        if label in plot_labels
    ]
    da_label_num_studies = sorted(
        da_label_num_studies,
        key=lambda x: label_to_index[x[0]]
    )
    df_label_num_studies = pandas.DataFrame(
        data=da_label_num_studies,
        columns=["Label", "Number of studies"]
    )
    
 
    sns.set(style="whitegrid")
    ax = sns.plt.axes()
    fig, axarr = plt.subplots(
        1,
        2,
        sharey=True,
        #figsize=(45, 1.0 + (0.2 * len(da_label_per_sample_recall)))
        figsize=(11, 1.0 + (0.2 * len(da_label_per_sample_recall)))
    )

    vis_lib.single_var_strip_plot(
        df_label_per_sample_recall,
        "Per-sample recall",
        "Label",
        x_lims=(0,1),
        ax=axarr[0]
    )

    """
    vis_lib.single_var_strip_plot(
        df_label_per_study_recall,
        "Per-study recall",
        "Label",
        x_lims=(0,1),
        ax=axarr[1]
    )
    """

    vis_lib.single_var_strip_plot(
        df_label_per_sample_prec,
        "Per-sample precision",
        "Label",
        x_lims=(0,1),
        ax=axarr[1]
    )

    """
    vis_lib.single_var_strip_plot(
        df_label_per_study_prec,
        "Per-study precision",
        "Label",
        x_lims=(0,1),
        ax=axarr[3]
    )

    vis_lib.single_var_strip_plot(
        df_label_num_samples,
        "Number of samples",
        "Label",
        ax=axarr[0]
    )

    vis_lib.single_var_strip_plot(
        df_label_num_studies,
        "Number of studies",
        "Label",
        ax=axarr[1]
    )
    """

    out_f = join(out_dir, "label_to_precision_recall_per_sample_per_study.pdf")
    sns.plt.tight_layout()
    fig.savefig(out_f, format='pdf', dpi=1000)

def plot_confusion_like_matrix(
    sample_to_labels,
    sample_to_predictions,
    out_dir,
    plot_labels=None
    ):

    label_to_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_samples[label].add(sample)

    true_label_to_pred_label_to_count = defaultdict(lambda: defaultdict(lambda: 0))
    for sample in sample_to_labels:
        labels = sample_to_labels[sample]
        predictions = sample_to_predictions[sample]
        for label in labels:
            for prediction in predictions:
                if label in plot_labels and prediction in plot_labels:
                    true_label_to_pred_label_to_count[label][prediction] += 1

    true_label_to_pred_label_to_fraction = defaultdict(lambda: defaultdict(lambda: 0))
    for label, pred_to_count in true_label_to_pred_label_to_count.iteritems():
        for pred, count in pred_to_count.iteritems():
            true_label_to_pred_label_to_fraction[label][pred] = float(count)/len(label_to_samples[label])

    cm = []
    labels_order = list(true_label_to_pred_label_to_fraction.keys())
    for label in labels_order:
        row = []
        for pred in labels_order:
            frac = true_label_to_pred_label_to_fraction[label][pred]
            row.append(frac)
        cm.append(row)

    fig, ax = plt.subplots(
        1,
        1,
#        sharey=True,
        figsize=(4.0 + (0.2 * len(labels_order)), 4.0 + (0.2 * len(labels_order)))
    )

    ylabels = [ONT_ID_TO_OG['17'].id_to_term[x].name for x in labels_order]
    ax = sns.heatmap(cm, fmt='', cbar=False, ax=ax)

#    ax.set_xticklabels(["" for x in ylabels])
    ax.set_xticklabels(ylabels)
    ax.set_yticklabels(ylabels)
    sns.plt.yticks(rotation=0)
    sns.plt.xticks(rotation=90)

    """
    fig = ax.get_figure()
    fig.set_size_inches(7, 7)
    #fig.savefig("confusion_matrix.png", bbox_inches='tight', dpi=400)
    fig.savefig("confusion_matrix.eps", format='eps', bbox_inches='tight', dpi=1200)

    sns.plt.show()
    """

    """ 
    da = []
    for label, pred_to_frac in true_label_to_pred_label_to_fraction.iteritems():
        for pred, frac in pred_to_frac.iteritems():
           da.append((label, pred, frac))
    df = pandas.DataFrame(
        data=da,
        columns=["True label", "Predicted label"],
    )
    """
    out_f = join(out_dir, "confusion_like_matrix.pdf")
    fig.savefig(out_f, format='pdf', dpi=1000)


if __name__ == "__main__":
    main()


