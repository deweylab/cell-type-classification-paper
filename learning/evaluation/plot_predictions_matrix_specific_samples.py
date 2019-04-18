###########################################################################################
#   Evaluate the output of a leave-study-out cross-validation experiment. More
#   specifically, this script analyzes the raw confidence scores output. There is no
#   summarization. Rather, this script shows predictions at the level of individual 
#   experiments. To make the plots smaller, a study argument can be supplied to restrict
#   the evaluation to only experiments in a given study.
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
from matplotlib import colors as mc
import pandas
import sklearn
from sklearn import metrics
import subprocess
import math
import numpy as np

sys.path.append("/ua/mnbernstein/projects/vis_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping")

import graph_lib
from graph_lib import graph
import project_data_on_ontology as pdoo
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import vis_lib as vl
import compute_metrics as cm
import the_ontology


PLOTS_BY_STUDY = True

def main():
    usage = "usage: %prog <options> <environment dir> <experiment list name> <cross-validation config name>"
    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--plot_study", help="Study for which to only plot experiments in that study")
    parser.add_option("-u", "--dont_enforce", action="store_true", help="TODO")
    parser.add_option("-o", "--out_dir", help="Directory in which to write the output")
    (options, args) = parser.parse_args()
    
    env_dir = args[0]
    exp_list_name = args[1]
    results_f = args[2]
    plot_exps_f = args[3]
    plot_study = options.plot_study
    out_dir = options.out_dir
    enforce_same_labels = not options.dont_enforce

    og = the_ontology.the_ontology()

    data_info_f = join(
        env_dir, 
        "data", 
        "data_set_metadata.json"
    )
    labels_f = join(
        env_dir, 
        "data", 
        "experiment_lists", 
        exp_list_name, 
        "labelling.json"
    )

    _run_cmd("mkdir -p %s" % out_dir)

    with open(plot_exps_f, 'r') as f:
        plot_exps = json.load(f)

    with open(results_f, 'r') as f:
        results = json.load(f)
    exp_to_label_to_conf = results['predictions']

    with open(data_info_f, 'r') as f:
        exp_to_info = json.load(f)
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)

    # load labellings 
    jsonable_graph = labels_data['label_graph']
    jsonable_exp_to_labels = labels_data['labelling']
    label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    exp_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_exp_to_labels)

    # compute the labels that we want to compute metrics for
    include_labels = cm.compute_included_labels(
        exp_to_label_to_conf,
        exp_to_labels,
        exp_to_info,
        og
    )
    print "Include labels: %s" % include_labels

    # convert label strings to labels
    exp_to_label_to_conf_new = defaultdict(lambda: {})
    for exp in exp_to_label_to_conf:
        for label_str, conf in exp_to_label_to_conf[exp].iteritems():
            label = pdoo.convert_node_from_str(label_str)
            exp_to_label_to_conf_new[exp][label] = conf
    exp_to_label_to_conf = exp_to_label_to_conf_new



    if enforce_same_labels:
        cm.validate_predictions(
            exp_to_label_to_conf,
            include_labels
        )
    else:
        # Compute the labels for which we will compute metrics over.
        # This label set is simply the set of labels for which we have
        # predictions for every sample
        ref_sample = sorted(exp_to_label_to_conf.keys())[0]
        include_labels = set(exp_to_label_to_conf[ref_sample].keys())
        for label_to_conf in exp_to_label_to_conf.values():
            include_labels = include_labels & set(label_to_conf.keys())

    label_to_name = {
        x: pdoo.convert_node_to_name(x, og)
        for x in label_graph.get_all_nodes()
    }

    num_labels = len(include_labels)

    if options.plot_study:
        plot_studies = [options.plot_study]
    else:
        plot_studies = list(set([
            exp_to_info[exp]['study_accession']
            for exp in exp_to_label_to_conf
        ]))

    exp_to_ms_labels = {
        exp: label_graph.most_specific_nodes(labels)
        for exp, labels in exp_to_labels.iteritems()
    }

    
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4.0 + (0.15 * len(include_labels)), 4.0)
    )
    plot_predictions_matrix( 
        exp_to_ms_labels,
        exp_to_label_to_conf, 
        label_to_name,
        label_graph,
        og,
        ax,
        plot_labels=include_labels,
        plot_exps=plot_exps
    )
    out_f = join(out_dir, "predictions_matrix")
    fig.savefig("%s.pdf" % out_f, format='pdf', bbox_inches='tight', dpi=1000)
    fig.savefig("%s.eps" % out_f, format='eps', bbox_inches='tight', dpi=1000)
   
 

def plot_predictions_matrix(
        exp_to_labels,
        exp_to_label_to_conf,
        label_to_name,
        label_graph,
        og,
        ax,
        plot_exps,
        plot_labels=None,
        plot_xtick_labels=True
    ):
    plot_exps_order = plot_exps # TODO this is annoying

    n_bin = 1000
    greens_cm = mc.LinearSegmentedColormap.from_list(
        'nice_greens', 
        ['#ffffff', vl.NICE_GREEN], 
        N=n_bin
    )
    blues_cm = mc.LinearSegmentedColormap.from_list(
        'nice_blues', 
        ['#ffffff', vl.NICE_BLUE], 
        N=n_bin
    )
    reds_cm = mc.LinearSegmentedColormap.from_list(
        'nice_reds', 
        ['#ffffff', vl.NICE_RED], 
        N=n_bin
    )
    purples_cm = mc.LinearSegmentedColormap.from_list( 
        'nice_purples', 
        ['#ffffff', vl.NICE_PURPLE], 
        N=n_bin
    )

    matrix = []
    mask_ms = []
    mask_anc = []
    mask_des = []
    mask_nei = []
    annot = []

    # Order the label columns according to their topological
    # order in the label DAG
    labels_order = graph.topological_sort(label_graph)
    labels_order = [
        label 
        for label in labels_order
        if label in plot_labels
    ]

    # Group the experiment-rows by their true labels
    label_set_to_exps = defaultdict(lambda: set())
    for exp in plot_exps:
        label_set = frozenset(exp_to_labels[exp])
        label_set_to_exps[label_set].add(exp)
    #plot_exps_order = []
    #for label_set, exps in label_set_to_exps.iteritems():
    #    plot_exps_order += sorted(exps)

    for exp in plot_exps_order:
        row = []
        mask_ms_row = []
        mask_anc_row = []
        mask_des_row = []
        mask_nei_row = []
        annot_row = []

        all_labels = set()
        for label in exp_to_labels[exp]:
            all_labels.update(label_graph.ancestor_nodes(label))
        ms_labels = label_graph.most_specific_nodes(all_labels)
        anc_labels = all_labels - ms_labels
        des_labels = set()
        for label in ms_labels:
            des_labels.update(label_graph.descendent_nodes(label))
        des_labels -= ms_labels

        anc_des_labels = set()
        for label in des_labels:
            anc_des_labels.update(label_graph.ancestor_nodes(label))
        anc_des_labels -= all_labels
        anc_des_labels -= des_labels

        for label in labels_order:
            conf = exp_to_label_to_conf[exp][label]
            if label in ms_labels:
                #print "Experiment %s predicted most specifically as %s" % (exp, label)
                mask_ms_row.append(False)
                annot_row.append('X')
            else:
                mask_ms_row.append(True)
                annot_row.append('')
            if label in anc_labels:
                mask_anc_row.append(False)
            else:
                mask_anc_row.append(True)
            if label in des_labels or label in anc_des_labels:
                mask_des_row.append(False)
            else:
                mask_des_row.append(True)
            if label not in ms_labels and label not in anc_labels and label not in des_labels and label not in anc_des_labels:
                mask_nei_row.append(False)
            else:
                mask_nei_row.append(True)
            row.append(conf)
        annot.append(annot_row)
        mask_ms.append(mask_ms_row)
        mask_anc.append(mask_anc_row)
        mask_des.append(mask_des_row)
        mask_nei.append(mask_nei_row)
        matrix.append(row)
    annot = np.array(annot)
#    fig, ax = plt.subplots(
#        1,
#        1,
##        sharey=True,
#        figsize=(4.0 + (0.2 * len(labels_order)), 4.0 + (0.2 * len(plot_exps_order)))
#    )

    ylabels = [
        label_to_name[x]
        for x in labels_order
    ]

    ax = sns.heatmap(matrix, mask=np.asarray(mask_des), fmt='', cmap=purples_cm, cbar=False, ax=ax, vmin=0.0, vmax=1.0, linewidths=0.3, linecolor='#b7b7b7')
    ax = sns.heatmap(matrix, mask=np.asarray(mask_anc), fmt='', cmap=blues_cm, cbar=False, ax=ax, vmin=0.0, vmax=1.0, linewidths=0.3, linecolor='#b7b7b7')
    ax = sns.heatmap(matrix, mask=np.asarray(mask_nei), fmt='', cmap=reds_cm, cbar=False, ax=ax, vmin=0.0, vmax=1.0, linewidths=0.3, linecolor='#b7b7b7')
    ax = sns.heatmap(matrix, annot=annot, mask=np.asarray(mask_ms), fmt='', cmap=greens_cm, cbar=False, ax=ax, vmin=0.0, vmax=1.0, linewidths=0.3, linecolor='#b7b7b7')

#    ax.set_xticklabels(["" for x in ylabels])
    ax.set_yticklabels(plot_exps_order, rotation='horizontal')
    #plt.yticks(rotation=90)
    if plot_xtick_labels:
        ax.set_xticklabels(ylabels, rotation='vertical', fontsize=20)
    #plt.xticks(rotation=90)
#    fig.savefig(out_f, format='pdf', bbox_inches='tight', dpi=1000)


def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)
























def evaluate_hard_predictions(): # TODO Write the method signature

    sample_to_prediction = apply_prediction_threshold(
        sample_to_label_to_confidence, 
        thresh
    )


    # convert predictions
    sample_to_predictions = {
        sample: set([
            pdoo.convert_node_from_str(x)
            for x in pred  
        ])
        for sample, pred in sample_to_predictions.iteritems()
    }

    # assure prediction samples are same as those labelled
    exp_to_labels = {k:v for k,v in exp_to_labels.iteritems() if k in sample_to_predictions}
    assert frozenset(sample_to_predictions.keys()) == frozenset(exp_to_labels.keys())

    print "TOTAL SAMPLES: %s" % len(sample_to_predictions)

    exp_to_info = data_info['sample_info']
    sample_accs_w_data = data_info['samples_with_recount_data']

    # Map study to samples in the cross-validation set on 
    # which predictions were made on
    study_to_samples = defaultdict(lambda: set())
    included_studies = set()
    for data_part in data_partitions:
        included_studies.update(data_partitions[data_part])
    for sample, sample_dat in exp_to_info.iteritems():
        study = sample_dat['study_accession']
        if sample in sample_to_predictions and study in included_studies:
            study_to_samples[study].add(sample)

    sample_accs = set()
    for study, samples in study_to_samples.iteritems():
        sample_accs.update(samples)

    label_to_studies = defaultdict(lambda: set())
    for sample, labels in exp_to_labels.iteritems():
        study = exp_to_info[sample]['study_accession']
        for label in labels:
            label_to_studies[label].add(study)
        
    sample_to_false_positives, sample_to_false_negatives, sample_to_hamming_loss = compute_per_sample_performance(
        sample_to_predictions,
        exp_to_labels
    )

    label_to_per_sample_recall = per_label_recall(
        sample_to_predictions,
        exp_to_labels
    )

    print "LABEL TO PER SAMPLE RECALL"
    print label_to_per_sample_recall

    label_to_per_sample_prec = per_label_precision(
        sample_to_predictions,
        exp_to_labels
    )

    # Compute the per-sample 'weight'. A sample's weight is
    # 1/num_samples where num_samples is number of samples
    # in that sample's study    
    sample_to_weight_recall = {}
    for study, samples in study_to_samples.iteritems():
        for sample in samples:
            sample_to_weight_recall[sample] = 1.0/len(samples)



    # TODO MAKE SURE THIS IS WORKING  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the per-sample 'weight' for each study to be 
    # used in the calculation of precision for the first 
    # method.
    label_to_sample_to_weight_prec = defaultdict(lambda: {})
    pred_label_to_studies = defaultdict(lambda: set())
    for sample, preds in sample_to_predictions.iteritems():
        study = exp_to_info[sample]['study_accession']
        for label in preds:
            pred_label_to_studies[label].add(study)
    pred_label_to_samples = defaultdict(lambda: set())
    for sample, preds in sample_to_predictions.iteritems():
        for label in preds:
            pred_label_to_samples[label].add(sample)
    for label in pred_label_to_samples:
        for study in pred_label_to_studies[label]:
            samples_of_label_in_study = set(pred_label_to_samples[label]) & set(study_to_samples[study])
            for sample in samples_of_label_in_study:
                label_to_sample_to_weight_prec[label][sample] = 1.0/len(samples_of_label_in_study)
    # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    label_to_per_study_recall = per_label_recall(
        sample_to_predictions,
        exp_to_labels,
        sample_to_weight_recall
    )

    label_to_per_study_prec = per_label_precision(
        sample_to_predictions,
        exp_to_labels,
        label_to_sample_to_weight_prec
    )

    # Map each label to set of relavents samples and studies
    label_to_exps = defaultdict(lambda: set())
    label_to_studies = defaultdict(lambda: set())
    for sample, labels in exp_to_labels.iteritems():
        study = exp_to_info[sample]['study_accession']
        for label in labels:
            label_to_exps[label].add(sample)
            label_to_studies[label].add(study)

#    plot_study_to_avg_hamming_loss(
#        study_to_avg_hamming_loss,
#        study_to_samples,
#        out_dir
#    )

    # We only want to plot labels that are represented by
    # at least two studies in the cross-validation set. These
    # will be labels that at least have the chance of being 
    # called correctly
    plot_labels = set([
        x for x in label_to_studies
        if len(label_to_studies[x]) >= 2
    ])
   
 
    plot_label_to_precision_recall_per_sample_per_study(
        label_to_per_sample_recall,
        label_to_per_study_recall,
        label_to_per_sample_prec,
        label_to_per_study_prec,
        label_to_exps,
        label_to_studies,
        og,
        out_dir,
        plot_labels=plot_labels
    )

    """    
    plot_label_to_precision_recall_per_sample(
        label_to_per_sample_recall,
        label_to_per_study_recall,
        label_to_per_sample_prec,
        label_to_per_study_prec,
        label_to_exps,
        label_to_studies,
        og,
        out_dir,
        plot_labels=plot_labels
    )
    """

    print "Total number of studies: %d" % len(study_to_samples)
    print "Total number of samples: %d" % len(sample_accs)
    print "Total number of labels: %d" % len(label_to_exps)


def compute_per_sample_performance(
        sample_to_predictions,
        exp_to_labels
    ):
    sample_to_false_positives = defaultdict(lambda: set())
    sample_to_false_negatives = defaultdict(lambda: set())
    sample_to_hamming_loss = {}
    for sample, predictions in sample_to_predictions.iteritems():
        predictions = set(predictions)
        labels = set(exp_to_labels[sample])
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
        exp_to_labels,
        sample_to_weight=None
    ):

    if not sample_to_weight:
        sample_to_weight = defaultdict(lambda: 1)

    label_to_predicted_samples = defaultdict(lambda: set())
    for sample, predictions in sample_to_predictions.iteritems():
        for label in predictions:
            label_to_predicted_samples[label].add(sample)

    label_to_relavent_samples = defaultdict(lambda: set())
    for sample, labels in exp_to_labels.iteritems():
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
        exp_to_predictions,
        exp_to_labels,
        label_to_sample_to_weight=None
    ):

    if not label_to_sample_to_weight:
        label_to_sample_to_weight = defaultdict(lambda: defaultdict(lambda: 1))

    label_to_predicted_samples = defaultdict(lambda: set())
    for sample, predictions in exp_to_predictions.iteritems():
        for label in predictions:
            label_to_predicted_samples[label].add(sample)

    label_to_relavent_samples = defaultdict(lambda: set())
    for sample, labels in exp_to_labels.iteritems():
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
        label_to_exps,
        label_to_studies,
        og,
        out_dir,
        plot_labels=None
    ):

    if not plot_labels:
        plot_labels = set(label_to_exps.keys()) # Use all of the labels


    random_labels = set(
        random.sample(
            label_to_per_sample_recall.keys(), 
            len(label_to_per_sample_recall)/4
        )
    )

    da_label_per_sample_recall = [
        (og.id_to_term[label].name, recall)
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
        (og.id_to_term[label].name, prec)
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
        (og.id_to_term[label].name, len(num_samples))
        for label, num_samples in label_to_exps.iteritems()
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

    vl.single_var_strip_plot(
        df_label_per_sample_recall,
        "Recall",
        "Cell-type",
        x_lims=(0,1),
        ax=axarr[0]
    )

    vl.single_var_strip_plot(
        df_label_per_sample_prec,
        "Precision",
        "Cell-type",
        x_lims=(0,1),
        ax=axarr[1]
    )

    vl.single_var_strip_plot(
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
        label_to_exps,
        label_to_studies,
        og,
        out_dir,
        plot_labels=None
    ):

    def label_to_name(label):
        print "THE LABEL IS %s" % label
        ms_terms = ontology_graph.most_specific_terms(label, og)
        ms_term = sorted(ms_terms)[0]
        name = og.id_to_term[ms_term].name
        return "%s (%d)" % (name, len(label))

    if not plot_labels:
        plot_labels = set(label_to_exps.keys()) # Use all of the labels
    
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
        for label, num_samples in label_to_exps.iteritems()
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

    vl.single_var_strip_plot(
        df_label_per_sample_recall,
        "Per-sample recall",
        "Label",
        x_lims=(0,1),
        ax=axarr[0]
    )

    """
    vl.single_var_strip_plot(
        df_label_per_study_recall,
        "Per-study recall",
        "Label",
        x_lims=(0,1),
        ax=axarr[1]
    )
    """

    vl.single_var_strip_plot(
        df_label_per_sample_prec,
        "Per-sample precision",
        "Label",
        x_lims=(0,1),
        ax=axarr[1]
    )

    """
    vl.single_var_strip_plot(
        df_label_per_study_prec,
        "Per-study precision",
        "Label",
        x_lims=(0,1),
        ax=axarr[3]
    )

    vl.single_var_strip_plot(
        df_label_num_samples,
        "Number of samples",
        "Label",
        ax=axarr[0]
    )

    vl.single_var_strip_plot(
        df_label_num_studies,
        "Number of studies",
        "Label",
        ax=axarr[1]
    )
    """

    out_f = join(out_dir, "label_to_precision_recall_per_sample_per_study.pdf")
    sns.plt.tight_layout()
    fig.savefig(out_f, format='pdf', dpi=1000)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n+1):
        yield l[i:i + n + 1]

if __name__ == "__main__":
    main()


