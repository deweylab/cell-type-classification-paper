#######################################################################################
#   Craetes plots for visualizing differences between precision and recall 
#   across terms between two learning algorithms
#######################################################################################

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
import matplotlib.patches as mpatches
import pandas
import math
import subprocess
import colour
from colour import Color
import numpy as np
import scipy
from scipy.stats import wilcoxon

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import project_data_on_ontology as pdoo
import the_ontology
from map_sra_to_ontology import ontology_graph
import vis_lib
from vis_lib import vis_lib as vl
import compute_metrics as cm

PHENOTYPING_ROOT = "/tier2/deweylab/mnbernstein/phenotyping_environments"

BLACKLIST_TERMS = set([
    "CL:0000010",   # cultured cell
    "CL:0000578",   # experimentally modified cell in vitro
    "CL:0001034"    # cell in vitro
])

def main():
    usage = "usage: %prog <options> <| delimited results files>, <| delimited method names>"
    parser = OptionParser()
    parser.add_option("-o", "--out_dir", help="Directory in which to write the output. If it doesn't exist, create the directory.")
    parser.add_option("-l", "--label_graph", help="The label graph file")
    parser.add_option("-g", "--color_group_ids", help="| delimited list of numbers indicating the groupings of methods to visualize the with the same color")
    parser.add_option("-c", "--conservative_metrics", action="store_true", help="Compute conservative metrics")
    (options, args) = parser.parse_args()

    # Parse the input
    #env_dir = args[0]
    #exp_list_name = args[1]
    config_f = args[0]
    #config_names = args[2].split('|')
    #method_names = args[3].split('|')

    # Load the ontology
    og = the_ontology.the_ontology()

    # Load the results
    #results_list = []
    #for config_name in config_names:
    #    results_f = join(
    #        env_dir,
    #        'data',
    #        'experiment_lists',
    #        exp_list_name,
    #        'leave_study_out_cv_results',
    #        config_name,
    #        'leave_study_out_cv_results.json' 
    #    )
    #    with open(results_f, 'r') as f:
    #        results_list.append(json.load(f))

    with open(config_f, 'r') as f:
        config_data = json.load(f)
    env_dir = config_data['env_dir']
    exp_list_name = config_data['exp_list_name']
    label_graph_f = config_data['test_label_graph_file']
    result_fs = config_data['result_files']
    method_names = config_data['method_names']
    indiv_plot_indices = config_data['boxplot_individuals']
    indiv_plot_names = config_data['boxplot_individual_names']

    matrix_scaffold = []
    for r in indiv_plot_indices:
        row = []
        for c in indiv_plot_indices:
           row.append((r,c))
        matrix_scaffold.append(row)

    out_dir = config_data['output_dir']
    if options.color_group_ids:
        color_group_ids = options.color_group_ids.split('|')
    else:
        color_group_ids = list(range(len(indiv_plot_names)))

    conservative_metrics = options.conservative_metrics

    boxplot_pairs = None
    if "boxplot_pairs" in config_data:
        boxplot_pairs = config_data['boxplot_pairs']
        paired_names = config_data['boxplot_pair_names']
        pair_names = config_data['pair_names'] 
    

    results_list = []
    for results_f in result_fs:
        try:
            with open(results_f, 'r') as f:
                results_list.append(json.load(f))
        except:
            raise Exception("Error reading results from %s" % results_f)

    sample_to_label_to_confidences = [
        results['predictions']
        for results in results_list
    ]

    # Make sure these are all results from the same data set
    #env_dir_ref = results_list[0]['data_set']['environment']
    #exp_list_name_ref = results_list[0]['data_set']['experiment_list_name']
    samples_ref = frozenset(sample_to_label_to_confidences[0].keys())
    for results, sample_to_label_to_confidence in zip(results_list, sample_to_label_to_confidences):
    #    assert results['data_set']['environment'] == env_dir_ref
    #    assert results['data_set']['experiment_list_name'] == exp_list_name_ref
        assert frozenset(sample_to_label_to_confidence.keys()) == samples_ref

    # Load the label graph
    #label_graph_f = join(
    #    env_dir,
    #    'data',
    #    'experiment_lists',
    #    exp_list_name,
    #    'labelling.json'
    #)
    with open(label_graph_f, 'r') as f:
        label_data = json.load(f)
    jsonable_graph = label_data['label_graph']
    label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    jsonable_sample_to_labels = label_data['labelling']
    sample_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_sample_to_labels)

    # Load the dataset metadata
    #env = results_list[0]['data_set']['environment']
    #exp_list_name = results_list[0]['data_set']['experiment_list_name']
    data_info_f = join(env_dir, "data", "data_set_metadata.json")
    with open(data_info_f, 'r') as f:
        sample_to_info = json.load(f)

    # Create the output directory
    _run_cmd("mkdir -p %s" % out_dir)

    # Convert label strings to labels
    sample_to_label_to_confidences_new = []
    for sample_to_label_to_confidence in sample_to_label_to_confidences:
        sample_to_label_to_confidence_new = defaultdict(lambda: {})
        for sample in sample_to_label_to_confidence:
            for label_str, conf in sample_to_label_to_confidence[sample].iteritems():
                label = pdoo.convert_node_from_str(label_str)
                sample_to_label_to_confidence_new[sample][label] = conf
        sample_to_label_to_confidences_new.append(sample_to_label_to_confidence_new)
    sample_to_label_to_confidences = sample_to_label_to_confidences_new

    # Compute the labels that are assigned to every
    # sample. These are trivial labels for which we don't
    # want to compute metrics
    trivial_labels = set(sample_to_labels[list(sample_to_labels.keys())[0]])
    for labels in sample_to_labels.values():
        trivial_labels = trivial_labels & set(labels)

    label_to_name = {
        x: pdoo.convert_node_to_name(x, og)
        for x in label_graph.get_all_nodes()
    } 

    # Which samples we should plot metrics for
    include_labels = cm.compute_included_labels(
        sample_to_label_to_confidences[0],
        sample_to_labels,
        sample_to_info,
        og
    )

    # Only compute metrics for the labels for which every 
    # experiment has a prediction for that label
    #for sample, label_to_confidence in sample_to_label_to_confidence_1.iteritems():
    #    labels = set(label_to_confidence.keys())
    #    include_labels = include_labels & labels
    #for sample, label_to_confidence in sample_to_label_to_confidence_2.iteritems():
    #    labels = set(label_to_confidence.keys())
    #    include_labels = include_labels & labels
    #print "Include labels: %s" % include_labels

    enforce_same_labels = False
    if enforce_same_labels:
        for sample_to_label_to_confidence in sample_to_label_to_confidences:
            cm.validate_predictions(
                sample_to_label_to_confidence,
                include_labels
            )
    else:
        # Compute the labels for which we will compute metrics over.
        # This label set is simply the set of labels for which we have
        # predictions for every sample
        ref_sample = sorted(sample_to_label_to_confidences[0].keys())[0]
        include_labels = set(sample_to_label_to_confidences[0][ref_sample].keys())
        for sample_to_label_to_confidence in sample_to_label_to_confidences:
            for label_to_confidence in sample_to_label_to_confidence.values():
                include_labels = include_labels & set(label_to_confidence.keys())

    samples_w_preds = list(sample_to_label_to_confidences[0].keys())
    print "%d samples with predictions" % len(samples_w_preds)
    print "Each has %d labels" % len(include_labels)

    # Compute all of the metrics
    label_to_aucs = []
    label_to_avg_precisions = []
    label_to_pr_curves = []
    global_pr_curves = []
    global_avg_precisions = []
    for sample_to_label_to_confidence in sample_to_label_to_confidences:
        if conservative_metrics:
            r = cm.compute_conservative_metrics(
                sample_to_label_to_confidence,
                sample_to_labels,
                label_graph,
                label_to_name,
                restrict_to_labels=include_labels,
                og=og
            )
        else:
            r = cm.compute_metrics(
                sample_to_label_to_confidence,
                sample_to_labels,
                label_graph,
                restrict_to_labels=include_labels,
                og=og
            )
        label_to_auc = r[0]
        label_to_avg_precision = r[1]
        label_to_pr_curve = r[2]
        global_pr_curve = r[3]
        global_avg_precision = r[4]
        label_to_aucs.append(label_to_auc)
        label_to_avg_precisions.append(label_to_avg_precision)
        label_to_pr_curves.append(label_to_pr_curve)
        global_pr_curves.append(global_pr_curve)
        global_avg_precisions.append(global_avg_precision)


    for label in BLACKLIST_TERMS:
        try:
            del label_to_avg_precision[label]
            del label_to_pr_curves[label]
            del label_to_aucs[label]
        except:
            pass   
 
    # Convert all the keys to the label name
    label_name_to_aucs = [
        convert_keys_to_label_name(label_to_auc, label_to_name)
        for label_to_auc in label_to_aucs
    ]
    label_name_to_avg_precisions = [
        convert_keys_to_label_name(label_to_avg_precision, label_to_name)
        for label_to_avg_precision in label_to_avg_precisions
    ]
    label_name_to_pr_curves = [
        convert_keys_to_label_name(label_to_pr_curve, label_to_name)
        for label_to_pr_curve in label_to_pr_curves
    ]

    label_name_to_max_recall_at_prec_threshs = [
        label_to_max_recall_at_prec_thresh(
            label_to_pr_curve,
            0.9
        )
        for label_to_pr_curve in label_to_pr_curves
    ]

    draw_collapsed_ontology_w_pr_curves(
        sample_to_labels,
        label_graph,
        label_to_name,
        [label_to_pr_curves[x] for x in indiv_plot_indices],
        [label_to_avg_precisions[x] for x in indiv_plot_indices],
        indiv_plot_names,
        color_group_ids,
        out_dir
    )
    draw_global_pr_curves(indiv_plot_names, [global_pr_curves[x] for x in indiv_plot_indices], out_dir)

    indiv_plot_indices = config_data['boxplot_individuals']
    indiv_plot_names = config_data['boxplot_individual_names']

    out_f_prefix = join(out_dir, "win_diff_avg_prec_heatmap")
    draw_comparison_heatmap(indiv_plot_names, matrix_scaffold, label_name_to_avg_precisions, out_f_prefix)

    out_f_prefix = join(out_dir, "win_diff_achievable_recall_heatmap")
    draw_comparison_heatmap(indiv_plot_names, matrix_scaffold, label_name_to_max_recall_at_prec_threshs, out_f_prefix)
    
    out_f_prefix = join(out_dir, "avg_prec_boxplot")
    draw_boxplot(
        indiv_plot_names, 
        [label_name_to_avg_precisions[x] for x in indiv_plot_indices], 
        'Avg. precision', 
        out_f_prefix
    )
    
    out_f_prefix = join(out_dir, "achievable_recall_boxplot")
    draw_boxplot(
        indiv_plot_names, 
        [label_name_to_max_recall_at_prec_threshs[x] for x in indiv_plot_indices], 
        'Achievable recall at 0.9 precision', 
        out_f_prefix
    )

    if boxplot_pairs:
        paired_label_name_to_avg_precisions = []
        for pair in boxplot_pairs:
            paired_label_name_to_avg_precisions.append((
                label_name_to_avg_precisions[pair[0]],
                label_name_to_avg_precisions[pair[1]]
            ))   
        out_f_prefix = join(out_dir, "paired_avg_prec_boxplot")    
        draw_paired_boxplot(paired_names, pair_names, paired_label_name_to_avg_precisions, 'Avg. precision', out_f_prefix)

        paired_label_name_to_max_recall_at_prec_thresh = []
        for pair in boxplot_pairs:
            paired_label_name_to_max_recall_at_prec_thresh.append((
                label_name_to_max_recall_at_prec_threshs[pair[0]],
                label_name_to_max_recall_at_prec_threshs[pair[1]]
            ))
        out_f_prefix = join(out_dir, "paired_achievable_recall_boxplot")
        draw_paired_boxplot(paired_names, pair_names, paired_label_name_to_max_recall_at_prec_thresh, 'Achievable recall at 0.9 precision', out_f_prefix)
 
def convert_keys_to_label_name(label_to_value, label_to_name):
    return {
        label_to_name[k]: v
        for k,v in label_to_value.iteritems()
    }


def _adjust_pr_curve_for_plot(precisions, recalls):
    new_precisions = [x for x in precisions]
    new_recalls = [x for x in recalls]
    prec_recs = zip(precisions, recalls)
    n_inserted = 0
    for i in range(1,len(prec_recs)):
        prec = prec_recs[i][0]
        rec = prec_recs[i][1]
        last_prec = prec_recs[i-1][0]
        last_rec = prec_recs[i-1][1]
        if rec > last_rec and prec < last_prec:
            #print "Found: (%f,%f) --> (%f,%f). Inserting: (%f, %f) at %d" % (last_rec, last_prec, rec, prec, last_rec, prec) 
            new_precisions.insert(i+n_inserted, prec)
            new_recalls.insert(i+n_inserted, last_rec)
            n_inserted += 1
    print "n inserted: %d" % n_inserted
    return new_precisions, new_recalls

def label_to_max_recall_at_prec_thresh(
        label_to_pr_curve,
        thresh
    ):
    label_to_achievable_rec = {}
    for label, pr_curve in label_to_pr_curve.iteritems():
        achievable_recs = [0.0]
        for prec, rec in zip(pr_curve[0], pr_curve[1]):
            if prec >= thresh:
                achievable_recs.append(rec)
        label_to_achievable_rec[label] =  max(achievable_recs)
    return label_to_achievable_rec

def draw_paired_boxplot(paired_names, pair_names, paired_label_name_to_vals, value_name, out_f_prefix):
    da = []
    for paired_name, paired_label_name_to_vals in zip(paired_names, paired_label_name_to_vals):
        for label, val in paired_label_name_to_vals[0].iteritems():
            da.append((
                paired_name,
                pair_names[0],
                label,
                val
            ))
        for label, val in paired_label_name_to_vals[1].iteritems():
            da.append((
                paired_name,
                pair_names[1],
                label,
                val
            ))
    df = pandas.DataFrame(
        data=da,
        columns = ['Method', 'Training strategy', 'Label', value_name]
    )
    print df
    sns.set_palette('colorblind')
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(0.75*len(paired_names), 2.5)
    )
    sns.boxplot(data=df, x='Method', y=value_name, hue='Training strategy', ax=ax)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    #sns.stripplot(data=df, x='Method', y=value_name, hue='Training strategy', ax=ax, color=".3", alpha=0.75)
    fig.savefig("%s.pdf" % out_f_prefix, format='pdf', dpi=1000, bbox_inches='tight')
    fig.savefig("%s.eps" % out_f_prefix, format='eps', dpi=1000, bbox_inches='tight')

def draw_boxplot(method_names, label_name_to_vals, value_name, out_f_prefix):
    da = []
    for method_name, label_to_val in zip(method_names, label_name_to_vals):
        for label, val in label_to_val.iteritems():
            da.append((
                method_name,
                label,
                val
            ))
    df = pandas.DataFrame(
        data=da,
        columns = ['Method', 'Label', value_name]
    )
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(0.55*len(method_names), 3)
    )
    sns.boxplot(data=df, x='Method', y=value_name, ax=ax)
    sns.stripplot(data=df, x='Method', y=value_name, ax=ax, color=".3", alpha=0.75, size=2)
    for method_name in method_names:
        if len(method_name) > 5:
            plt.xticks(rotation=90)
            break
    fig.savefig("%s.pdf" % out_f_prefix, format='pdf', dpi=1000, bbox_inches='tight')
    fig.savefig("%s.eps" % out_f_prefix, format='eps', dpi=1000, bbox_inches='tight') 


def draw_comparison_heatmap(method_names, matrix_scaffold, label_name_to_vals, out_f_prefix):
    print "Generating average precision comparison matrix..."
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(2.75, 2.75)
    )
    matrix = []
    annot = []
    all_vals = []
    stat_sig_mask = []
    diag_mask = []
    for scaffold_row in matrix_scaffold:
        row = []
        annot_row = []
        stat_sig_mask_row = []
        diag_mask_row = []
        for comparison_indices in scaffold_row:
            method_i = comparison_indices[0]
            method_j = comparison_indices[1]
            label_diffs = []
            for label in label_name_to_vals[method_i]:
                if label in BLACKLIST_TERMS:
                    continue
                i_val = label_name_to_vals[method_i][label]
                j_val = label_name_to_vals[method_j][label]
                diff = i_val - j_val
                label_diffs.append(diff)

            p_val = wilcoxon(label_diffs)[1]
            if p_val < 0.05:
                stat_sig_mask_row.append(False)
            else:
                stat_sig_mask_row.append(True)
                

            #median_diff = np.median(label_diffs)
            #row.append(median_diff)
            n_i_beat_j = len([x for x in label_diffs if x > 0 and abs(x) > 0.05])
            n_j_beat_i = len([x for x in label_diffs if x < 0 and abs(x) > 0.05])
            win_diff = n_i_beat_j - n_j_beat_i
            row.append(win_diff)
            #print "Median diff between %s and %s is %f" % (method_i, method_j, median_diff)
            #print "Median diff between %s and %s is %f" % (method_i, method_j, win_diff)
            #all_vals.append(median_diff)
            all_vals.append(win_diff)
            if method_i == method_j:
                diag_mask_row.append(False)
                n_best = 0
                for label in label_name_to_vals[method_i]:
                    if label in BLACKLIST_TERMS:
                        continue
                    val = label_name_to_vals[method_i][label]
                    is_best = True
                    for method_k, label_name_to_val in enumerate(label_name_to_vals):
                        if method_i == method_k:
                            continue
                        if val <= label_name_to_val[label] or abs(val - label_name_to_val[label]) < 0.05:
                            is_best = False
                            break
                    if is_best:
                        n_best += 1
                annot_row.append("%d" % n_best)
            else:
                annot_row.append("%d" % win_diff)
                diag_mask_row.append(True)
        matrix.append(row)
        annot.append(annot_row)
        stat_sig_mask.append(stat_sig_mask_row)
        diag_mask.append(diag_mask_row)
    #mask = np.zeros_like(matrix)
    #mask[np.triu_indices_from(mask)] = True
    #for i in range(len(mask)):
    #    mask[i][i] = True
    with sns.axes_style("white"):
        cmap = sns.diverging_palette(0,255,sep=30, as_cmap=True)
        print "The matrix is: %s" % matrix
        ax = sns.heatmap(
            matrix, 
            mask=np.array(stat_sig_mask), 
            vmin=-max(all_vals), 
            vmax=max(all_vals), 
            cbar=False, 
            cmap=cmap, 
            annot=np.array(annot),
            fmt='',
            yticklabels=method_names,
            xticklabels=method_names,
            annot_kws={"weight": "bold"}
        )
        ax = sns.heatmap(
            matrix,
            mask=np.array([
                [not x for x in row]
                for row in stat_sig_mask
            ]),
            vmin=-max(all_vals),
            vmax=max(all_vals),
            cbar=False,
            cmap=cmap,
            annot=np.array(annot),
            fmt='',
            yticklabels=method_names,
            xticklabels=method_names,
            annot_kws={"fontsize": "8"}
        )
        dark_middle = sns.diverging_palette(255, 133, l=60, n=7, center="dark")
        ax = sns.heatmap(
            matrix,
            mask=np.array(diag_mask),
            vmin=-max(all_vals),
            vmax=max(all_vals),
            cbar=False,
            cmap=dark_middle,
            annot=np.array(annot),
            fmt='',
            yticklabels=method_names,
            xticklabels=method_names
        )
    fig.savefig("%s.pdf" % out_f_prefix, format='pdf', dpi=1000, bbox_inches='tight')
    fig.savefig("%s.eps" % out_f_prefix, format='eps', dpi=1000, bbox_inches='tight') 
    print "done."

def draw_comparison_stripplot_precision(
        label_name_to_avg_precision_1,
        label_name_to_avg_precision_2
    ):
    sns.set(style="whitegrid")
    fig, axarr = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10, 0.22*len(label_name_to_avg_precision_2))
    )

    vl.performance_comparison_stripplot(
        {
            method_1_name: label_name_to_avg_precision_1,
            method_2_name: label_name_to_avg_precision_2
        },
        "Method",
        "Label",
        "Average precision",
        ax=axarr[0]
    )

    da_num_studies = [
        (label, n_studies)
        for label, n_studies in label_name_to_num_studies.iteritems()
    ]
    df_num_studies = pandas.DataFrame(
        data=da_num_studies,
        columns=["Label", "Number of studies"]
    )
    vl.single_var_strip_plot(
        df_num_studies,
        "Number of studies",
        "Label",
        ax=axarr[1],
    )

    out_f = join(
        out_dir,
        "compare_strip.avg_prec.%s_VS_%s.pdf" % (
            method_1_name.replace(" ", "_"),
            method_2_name.replace(" ", "_")
        )
    )
    fig.savefig(out_f, format='pdf', dpi=1000, bbox_inches='tight')



def draw_comparison_stripplot_recall(
        label_name_to_recall_1,
        label_name_to_recall_2,
        out_dir
    ):
    sns.set(style="whitegrid")
    fig, axarr = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10, 0.22*len(label_name_to_recall_1))
    )

    vl.performance_comparison_stripplot(
        {
            method_1_name: label_name_to_recall_1,
            method_2_name: label_name_to_recall_2
        },
        "Method",
        "Label",
        "Recall",
        ax=axarr[0]
    )

    vl.single_var_strip_plot(
        df_num_studies,
        "Number of studies",
        "Label",
        ax=axarr[1],
    )

    out_f = join(
        out_dir,
        "compare_strip.recall.%s_VS_%s.pdf" % (
            method_1_name.replace(" ", "_"),
            method_2_name.replace(" ", "_")
        )
    )


def draw_grid_w_pr_curves(
        label_name_to_pr_curves,
        method_names,
        out_dir
    ):
    """
    precision recall curve
    """
    num_cols = 11
    num_rows = int(math.ceil(len(label_name_to_pr_curves[0]) / num_cols)) + 1

    print "Num columns: %d" % num_cols
    print "Num rows: %d" % num_rows

    sns.set(style="dark")
    fig, axarr = plt.subplots(
        num_rows,
        num_cols,
        sharey=True,
        sharex=True,
        figsize=(int(5.0 * num_cols), int(5.0 * num_rows)),
        squeeze=False
    )

    for label_i, label in enumerate(label_name_to_pr_curves[0]):
        row = int(math.floor(label_i / num_cols))
        col = label_i % num_cols
        print "Grabbing axis at row %d, col %d" % (row, col)
        ax = axarr[row][col]

        curves = []
        for curve_i, label_name_to_pr_curve in enumerate(label_name_to_pr_curves):
            pr = label_name_to_pr_curve[label]
            precisions = pr[0]
            recalls = pr[1]
            l, = ax.plot(
                recalls, 
                precisions, 
                color=vl.NICE_COLORS[curve_i], 
                lw=8
            )
            curves.append(l)
        ax.set_title(label)

    plt.legend(
        curves,
        method_names,
        loc='lower center',
        bbox_to_anchor=(0,-0.1,1,1),
        bbox_transform = plt.gcf().transFigure
    )
    plt.tight_layout()
    out_f = join(out_dir, "label_to_pr_curve.pdf")
    fig.savefig(out_f, format='pdf', bbox_inches='tight', dpi=200)



def draw_global_pr_curves(method_names, global_pr_curves, out_dir):
    fig, axarr = plt.subplots(
        1,
        1,
        figsize=(3.0, 3.0),
        squeeze=False
    )
    ax = axarr[0][0]
    for curve_i, pr in enumerate(global_pr_curves):
        precisions = pr[0]
        recalls = pr[1]
        precisions, recalls = _adjust_pr_curve_for_plot(precisions, recalls)
        ax.plot(
            recalls,
            precisions,
            color=vl.NICE_COLORS[curve_i],
            lw=1.5
            #where='pre'
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Joint-recall')
    ax.set_ylabel('Joint-precision')
    patches = [
        mpatches.Patch(
            color=vl.NICE_COLORS[method_i],
            label=method_names[method_i]
        )
        for method_i in range(len(method_names))
    ]
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        handles=patches
    )
    out_pref = join(out_dir, "pan_data_set_pr_curves")
    fig.savefig(
        "%s.pdf" % out_pref,
        format='pdf',
        bbox_inches='tight'
    )
    fig.savefig(
        "%s.eps" % out_pref,
        format='eps',
        bbox_inches='tight'
    )
 

def draw_collapsed_ontology_w_pr_curves(
        sample_to_labels,
        label_graph, 
        label_to_name,
        label_to_pr_curves, 
        label_to_avg_precisions,
        method_names,
        color_group_ids,
        out_dir
    ):
    """
        color_group_ids: a list of ID's indicating which group each 
            method belongs to. Methods in the same group should be 
            colored the same. For example, if the groupings are
            '0,0,1' this indicates the first two methods should be
            colored with the same color and the third method should
            get its own color.
    """

    assert len(color_group_ids) == len(method_names)

    tmp_dir = join(out_dir, "tmp_figs")
    _run_cmd("mkdir -p %s" % tmp_dir)

    source_to_targets = label_graph.source_to_targets
    target_to_sources = label_graph.target_to_sources

    label_to_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_samples[label].add(sample)
    label_to_num_samples = {
        k:len(v) 
        for k,v in label_to_samples.iteritems()
    }

    all_group_ids = sorted(list(set(color_group_ids)))
    group_id_to_color = {
        group_id: vl.NICE_COLORS[i]
        for i, group_id in enumerate(all_group_ids)
    }
    colors = [
        group_id_to_color[group_id]
        for group_id in color_group_ids
    ]
    
    method_to_color = {
        method_name: colors[method_i]
        for method_i, method_name in enumerate(method_names)
    }

    label_to_color = {}
    label_to_diff_avg_prec = {}
    label_to_fig = {}
    for label in label_to_pr_curves[0]:

        label_avg_precs = [
            label_to_avg_precision[label]
            for label_to_avg_precision in label_to_avg_precisions
        ]
        method_w_avg_precs = sorted(
            zip(method_names, label_avg_precs), 
            key=lambda x: x[1]
        )
       
        label_to_color[label] = method_to_color[method_w_avg_precs[-1][0]]
        print "Label %s" % label
        print "Best method: %f" % method_w_avg_precs[-1][1]
        print "Second best method: %f" % method_w_avg_precs[-2][1]
        print
        label_to_diff_avg_prec[label] = abs(method_w_avg_precs[-1][1] - method_w_avg_precs[-2][1])

        fig, axarr = plt.subplots(
            1,
            1,
            figsize=(3.0, 3.0),
            squeeze=False
        )
        for curve_i, label_to_pr_curve in enumerate(label_to_pr_curves):
            pr = label_to_pr_curve[label]
            precisions = pr[0]
            recalls = pr[1]
            precisions, recalls = _adjust_pr_curve_for_plot(precisions, recalls)
            axarr[0][0].plot(
                recalls,
                precisions,
                color='black',
                lw=8.5
                #where='pre'
            )
            axarr[0][0].plot(
                recalls, 
                precisions, 
                color=vl.NICE_COLORS[curve_i], 
                lw=8
                #where='pre'
            )
        title = "%s\n(n=%d)" % (label_to_name[label], len(label_to_samples[label]))

        """
        # Format the title
        if len(title) > 25:
            toks = title.split(" ")
            str_1 = ""
            str_2 = ""
            t_i = 0
            for t_i in range(len(toks)):
                if len(str_1) > 25:
                    break
                str_1 += " " + toks[t_i]
            for t_i in range(t_i, len(toks)):
                str_2 += " " + toks[t_i]
            title = "%s\n%s" % (str_1, str_2)
        """ 
        axarr[0][0].set_title(title, fontsize=16)
        axarr[0][0].set_xlim(0.0, 1.0)
        axarr[0][0].set_ylim(0.0, 1.0)
        out_f = join(tmp_dir, "%s.png" % label_to_name[label].replace(' ', '_').replace('/', '_'))
        fig.savefig(
            out_f, 
            format='png', 
            bbox_inches='tight', 
            dpi=200,
            transparent=True
        )
        label_to_fig[label] = out_f
    result_dot_str = _diff_dot(
        source_to_targets,
        label_to_name, 
        label_to_num_samples, 
        label_to_fig,
        label_to_color,
        label_to_diff_avg_prec 
    )
    dot_f = join(tmp_dir, "collapsed_ontology_by_name.dot")
    graph_out_f = join(out_dir, "pr_curves_on_graph.pdf")
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, graph_out_f))
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)


def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)


def _diff_dot(
        source_to_targets, 
        node_to_label, 
        node_to_num_samples, 
        node_to_image, 
        node_to_color, 
        node_to_color_intensity 
    ):
    max_samples = float(math.log(max(node_to_num_samples.values())+1.0))
    max_color_intensity = float(math.log(max(node_to_color_intensity.values())+1.0))

    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        if node not in node_to_image:
            continue
        intensity = math.log(node_to_color_intensity[node]+1.0) / max_color_intensity 

        # The algorithm used to set the color's luminosity: 
        #
        #  black        color  target    white
        #   |--------------|-----|---------|     luminosity     
        #  0.0                   |        1.0
        #                        |
        #                        |                    
        #                  |-----|---------|     intensity
        #                 1.0  intensity  0.0          
        
        # luminance corresponding to maximum intensity
        min_luminance = Color(node_to_color[node]).luminance 
        # difference of luminance we allow between pure white and the target color
        luminance_range = 1.0 - min_luminance 
        luminance = min_luminance + ((1.0 - intensity) * luminance_range)
        color_value = Color(node_to_color[node], luminance=luminance).hex_l
 
        pen_width = (math.log(node_to_num_samples[node]+1.0)/max_samples)*15
        g += '"%s" [style=filled,  fillcolor="%s", image="%s", label=""]\n' % (
            node_to_label[node],
            color_value,
            node_to_image[node]
        )
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            if source in node_to_image and target in node_to_image:
                g += '"%s" -> "%s"\n' % (
                    node_to_label[source],
                    node_to_label[target]
                )
    g += "}"
    return g


def draw_label_dendrogram(
        cluster_to_labels,
        cluster_to_children,
        label_to_name,
        out_dir
    ): 
    tmp_dir = join(out_dir, "tmp_figs")
    _run_cmd("mkdir -p %s" % tmp_dir)

    node_to_str = {}
    node_to_is_internal = {}
    for cluster, children in cluster_to_children.iteritems():
        if len(children) == 0:
            node_to_str[cluster] = label_to_name[list(cluster_to_labels[cluster])[0]]
            node_to_is_internal[cluster] = False
        else:
            node_to_str[cluster] = str(cluster)
            node_to_is_internal[cluster] = True
    dot_str = _dendrogram_dot(
        cluster_to_children,
        node_to_str,
        node_to_is_internal
    )
    dot_f = join(tmp_dir, "labels_dendrogram.dot")
    graph_out_f = join(out_dir, "labels_dendrogram.pdf")
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, graph_out_f))
    with open(dot_f, 'w') as f:
        f.write(dot_str)

    
def _dendrogram_dot(
        source_to_targets,
        node_to_label,
        node_to_is_internal
    ):
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for node in all_nodes:
        if node_to_is_internal[node]:
            color = '#%02x%02x%02x' % (0,0,0)
            g += '"%s" [style=filled,  fillcolor="%s", label=""]\n' % (
                node_to_label[node],
                color
            )
        else:
            g += '"%s"\n' % node_to_label[node]
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            g += '"%s" -> "%s"\n' % (
                node_to_label[source],
                node_to_label[target]
            )
    g += "}"
    return g


def draw_dendrogram_weighted_avg_prec_curve(
        clust_heights,
        weighted_avg_precs_1,
        weighted_avg_precs_2,
        method_1_name,
        method_2_name,
        out_dir
    ):

    patch_1 = mpatches.Patch(label=method_1_name, color=vl.NICE_RED)
    patch_2 = mpatches.Patch(label=method_2_name, color=vl.NICE_BLUE)
    fig, axarr = plt.subplots(
        1,
        1,
        figsize=(5.0, 3.0),
        squeeze=False
    )
    axarr[0][0].plot(clust_heights, weighted_avg_precs_1, color=vl.NICE_RED, lw=4)
    axarr[0][0].plot(clust_heights, weighted_avg_precs_2, color=vl.NICE_BLUE, lw=4)
    axarr[0][0].set_title("Cluster-weighted average-precision across dendrogram")
    axarr[0][0].set_ylim(0.0, 1.0)
    axarr[0][0].legend(handles=[patch_1, patch_2])
    out_f = join(out_dir, "avg_prec_dendrogram_curve.pdf")
    fig.savefig(
        out_f,
        format='pdf',
        bbox_inches='tight',
        dpi=100
    )




if __name__ == "__main__":
    main()


