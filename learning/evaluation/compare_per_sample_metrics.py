#######################################################################################
#   Craetes plots for visualizing differences between precision and recall 
#   across terms between two learning algorithms
#######################################################################################

# TODO THIS SCRIPT SHOULD PROBABLY TAKE THE PLACE OF compare_mult_methods.py 

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
import numpy as np
import string
import random

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import project_data_on_ontology as pdoo
import the_ontology
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import vis_lib
from vis_lib import vis_lib as vl
import compute_metrics as cm

REMOVE_LABELS = [
    frozenset(["CL:0000548"]),    # animal cell
    frozenset(["CL:0000255"]),    # eukaryotic cell
    frozenset(["CL:0000010"]),    # cultured cell
    frozenset(["CL:0000578"]),    # experimentally modified cell in vitro
    frozenset(["CL:0001034"])     # cell in vitro
]

def main():
    usage = "usage: %prog <options> <1st results file> <2nd results file> <name of 1st method> <name of 2nd method>"
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-a", "--thresh_1", type="float", help="Hard prediction threshold for method 1")
    parser.add_option("-b", "--thresh_2", type="float", help="Hard prediction threshold for method 2")
    parser.add_option("-o", "--out_dir", help="Directory in which to write the output. If it doesn't exist, create the directory.")
    parser.add_option("-c", "--conservative_metrics", action="store_true", help="Compute conservative version of these metrics")
    parser.add_option("-l", "--label_graph", help="The label graph file")
    (options, args) = parser.parse_args()

    config_f = args[0]
    conservative_metrics = options.conservative_metrics

    with open(config_f, 'r') as f:
        config = json.load(f)
    results_fs = config['result_files']
    method_names = config['boxplot_individual_names']
    plot_indices = config['boxplot_individuals']
    env_dir = config['env_dir']
    exp_list_name = config['exp_list_name']
    label_graph_f = config['test_label_graph_file']
    #train_label_graph_f = config['train_label_graph_file']
    out_dir = config['output_dir']
    pred_thresh_ranges = config['pred_thresh_ranges']

    #results_fs = args[0].split(",")
    #print "Results files: %s" % results_fs
    #method_names = args[1].split(",")
    pred_thresh = 0.5

    og = the_ontology.the_ontology()
 
    #label_graph_f = options.label_graph
    with open(label_graph_f, 'r') as f:
        label_data = json.load(f)    
    jsonable_graph = label_data['label_graph']
    label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    jsonable_sample_to_labels = label_data['labelling']
    sample_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_sample_to_labels)

    #with open(train_label_graph_f, 'r') as f:
    #    train_label_data = json.load(f)
    #train_jsonable_graph = train_label_data['label_graph']
    #train_label_graph = pdoo.graph_from_jsonable_dict(train_jsonable_graph)

    _run_cmd("mkdir -p %s" % out_dir)

    results = []
    for results_f in results_fs:
        with open(results_f, 'r') as f:
            results.append(json.load(f))

    sample_to_label_to_confidences = []
    for result in results:
        sample_to_label_to_confidences.append(
            result['predictions']
        )

    #reference_exp_list_name = results[0]['data_set']['experiment_list_name']
    reference_samples = frozenset(sample_to_label_to_confidences[0].keys()) 
    #for result in results:
    #    assert result['data_set']['environment'] == env_dir
    #    assert result['data_set']['experiment_list_name'] == reference_exp_list_name
    for sample_to_label_to_confidence in sample_to_label_to_confidences:
        assert frozenset(sample_to_label_to_confidence.keys()) == reference_samples
    
    data_info_f = join(env_dir, "data", "data_set_metadata.json")

    with open(data_info_f, 'r') as f:
        exp_to_info = json.load(f)

    sample_to_study = {
        exp: info['study_accession']
        for exp, info in exp_to_info.iteritems()
        if exp in reference_samples
    }
    study_to_samples = defaultdict(lambda: set())
    for sample, study in sample_to_study.iteritems():
        study_to_samples[study].add(sample)
    study_to_samples = dict(study_to_samples)

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

    trivial_labels = set(sample_to_labels[list(sample_to_labels.keys())[0]])
    for labels in sample_to_labels.values():
        trivial_labels = trivial_labels & set(labels)

    # data metrics
    label_to_num_studies = cm.data_metrics(
        sample_to_label_to_confidences[0].keys(),
        sample_to_labels,
        exp_to_info
    )

    label_to_name = {
        x: pdoo.convert_node_to_name(x, og)
        for x in label_graph.get_all_nodes()
    } 

    label_name_to_num_studies = convert_keys_to_label_name(
        label_to_num_studies, 
        label_to_name
    )

    # which samples we should plot metrics for
    include_labels = cm.compute_included_labels(
        sample_to_label_to_confidences[0],
        sample_to_labels,
        exp_to_info,
        og
    )

    for label in REMOVE_LABELS:
        try:
            include_labels.remove(label)
        except:
            pass 
    print "Include labels: %s" % include_labels


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
        #include_labels &= label_graph.get_all_nodes()

    #for sample_to_label_to_confidence in sample_to_label_to_confidences:
    #    cm.validate_predictions(
    #        sample_to_label_to_confidence,
    #        include_labels
    #    )

    samples_w_preds = list(sample_to_label_to_confidences[0].keys())
    print "%d samples with predictions" % len(samples_w_preds)
    print "Each has %d labels" % len(include_labels)
   
    # TODO this is somewhat hacky...
    sample_to_label_to_confidences = [sample_to_label_to_confidences[x] for x in plot_indices]
 
    label_to_aucs = []
    label_to_avg_precisions = []
    label_to_recalls = []
    label_to_precisions = []
    label_to_pr_curves = []
    hierarchical_pr_curves = []
    sw_hierarchical_pr_curves = []
    ms_hierarchical_pr_curves = []
    sw_ms_hierarchical_pr_curves = []
    for sample_to_label_to_confidence, pred_thresh_range  in zip(sample_to_label_to_confidences, pred_thresh_ranges):
        if conservative_metrics:
            r = cm.compute_conservative_per_sample_metrics(
                sample_to_label_to_confidence,
                sample_to_labels,
                label_graph,
                sample_to_study,
                study_to_samples,
                label_to_name=label_to_name,
                restrict_to_labels=include_labels,
                og=og,
                thresh_range=pred_thresh_range
            )
        else:
            r = cm.compute_per_sample_metrics(
                sample_to_label_to_confidence,
                sample_to_labels,
                label_graph,
                sample_to_study,
                study_to_samples,
                label_to_name=label_to_name,
                restrict_to_labels=include_labels,
                og=og,
                thresh_range=pred_thresh_range
            )
        hierarchical_pr_curve = r[2]
        sw_hierarchical_pr_curve = r[3]
        ms_hierarchical_pr_curve = r[4]
        sw_ms_hierarchical_pr_curve = r[5]
        hierarchical_pr_curves.append(hierarchical_pr_curve)
        sw_hierarchical_pr_curves.append(sw_hierarchical_pr_curve)
        ms_hierarchical_pr_curves.append(ms_hierarchical_pr_curve)
        sw_ms_hierarchical_pr_curves.append(sw_ms_hierarchical_pr_curve)


    # Convert all the keys to the label name
    label_name_to_aucs = []
    label_name_to_avg_precisions = []
    label_name_to_recalls = []
    label_name_to_precisions = []
    label_name_to_pr_curves = []
    for i in range(len(label_name_to_aucs)):
        label_name_to_aucs.append(
            convert_keys_to_label_name(label_to_aucs[i], label_to_name)
        )
        label_name_to_avg_precisions.append(
            convert_keys_to_label_name(label_to_avg_precisions[i], label_to_name)
        )
        label_name_to_recalls.append(
            convert_keys_to_label_name(label_to_recalls[i], label_to_name)
        )
        label_name_to_precisions.append(
            convert_keys_to_label_name(label_to_precisions[i], label_to_name)
        )
        label_name_to_pr_curves.append(
            convert_keys_to_label_name(label_to_pr_curves[i], label_to_name)
        )

    # TODO this should be the multiple method version
#    draw_collapsed_ontology_w_pr_curves(
#        sample_to_labels,
#        label_graph,
#        label_to_name,
#        label_to_pr_curve_1,
#        label_to_pr_curve_2,
#        label_to_avg_precision_1,
#        label_to_avg_precision_2,
#        out_dir
#    )

    sns.set_style("white")
    fig, ax = plt.subplots(
        2,
        2,
        sharey=True,
        sharex=True,
        figsize=(5.0, 4.0)
    )
    for method_i in range(len(method_names)):
        pr = hierarchical_pr_curves[method_i]
        sw_pr = sw_hierarchical_pr_curves[method_i]
        ms_pr = ms_hierarchical_pr_curves[method_i]
        sw_ms_pr = sw_ms_hierarchical_pr_curves[method_i]

        precisions = pr[0]
        recalls = pr[1]
        sw_precisions = sw_pr[0]
        sw_recalls = sw_pr[1]
        ms_precisions = ms_pr[0]
        ms_recalls = ms_pr[1]
        sw_ms_precisions = sw_ms_pr[0]
        sw_ms_recalls = sw_ms_pr[1]
        l, = ax[0][0].plot(recalls, precisions, color=vl.NICE_COLORS[method_i], lw=1.5)
        ax[0][0].set_title("Mean PR-curve")
        l, = ax[0][1].plot(sw_recalls, sw_precisions, color=vl.NICE_COLORS[method_i], lw=1.5)
        ax[0][1].set_title("Weighted-mean PR-curve")
        l, = ax[1][0].plot(ms_recalls, ms_precisions, color=vl.NICE_COLORS[method_i], lw=1.5)
        ax[1][0].set_title("Mean Specific-PR-curve")
        l, = ax[1][1].plot(sw_ms_recalls, sw_ms_precisions, color=vl.NICE_COLORS[method_i], lw=1.5)
        ax[1][1].set_title("Weighted-mean\nSpecific-PR-curve")
    ax[0][0].set_xlim((0.0, 1.0))
    ax[0][1].set_ylim((0.0, 1.0))
    ax[1][0].set_xlim((0.0, 1.0))
    ax[1][1].set_ylim((0.0, 1.0))
    patches = [
        mpatches.Patch(
            color=vl.NICE_COLORS[method_i], 
            label=method_names[method_i]
        )
        for method_i in range(len(method_names))
    ]
    ax[0][1].legend(
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        handles=patches
    ) 
    plt.tight_layout()
    fig.savefig(
        join(out_dir, "per_sample_hierarchical_metrics.pdf"), format='pdf', dpi=1000, bbox_inches='tight'
    )
    fig.savefig(
        join(out_dir, "per_sample_hierarchical_metrics.eps"), format='eps', dpi=1000, bbox_inches='tight'
    )


def convert_keys_to_label_name(label_to_value, label_to_name):
    return {
        label_to_name[k]: v
        for k,v in label_to_value.iteritems()
    }












def draw_grid_w_pr_curves(
        label_name_to_pr_curve_1,
        label_name_to_pr_curve_2,
        method_1_name,
        method_2_name,
        out_dir
    ):
    """
    precision recall curve
    """
    num_cols = 11
    num_rows = int(math.ceil(len(label_name_to_pr_curve_2) / num_cols)) + 1

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

    for label_i, label in enumerate(label_name_to_pr_curve_1):
        row = int(math.floor(label_i / num_cols))
        col = label_i % num_cols
        print "Grabbing axis at row %d, col %d" % (row, col)
        ax = axarr[row][col]

        pr_1 = label_name_to_pr_curve_1[label]
        precisions_1 = pr_1[0]
        recalls_1 = pr_1[1]

        pr_2 = label_name_to_pr_curve_2[label]
        precisions_2 = pr_2[0]
        recalls_2 = pr_2[1]

        l_1, = ax.plot(recalls_1, precisions_1, color=vl.NICE_RED, lw=8)
        l_2, = ax.plot(recalls_2, precisions_2, color=vl.NICE_BLUE, lw=8)
        ax.set_title(label)

    plt.legend(
        [l_1, l_2],
        [method_1_name, method_2_name],
        loc='lower center',
        bbox_to_anchor=(0,-0.1,1,1),
        bbox_transform = plt.gcf().transFigure
    )
    plt.tight_layout()
    out_f = join(out_dir, "label_to_pr_curve.pdf")
    fig.savefig(out_f, format='pdf', bbox_inches='tight', dpi=200)


def draw_grid_w_pr_curve_and_point(
        label_name_to_pr_curve_1,
        label_name_to_precision_2,
        out_dir
    ):
    """
    precision recall curve of method 1 with point from method 2
    """
    num_cols = 11
    num_rows = int(math.ceil(len(label_name_to_pr_curve_2) / num_cols)) + 1

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

    for label_i, label in enumerate(label_name_to_pr_curve_1):
        row = int(math.floor(label_i / num_cols))
        col = label_i % num_cols
        print "Grabbing axis at row %d, col %d" % (row, col)
        ax = axarr[row][col]

        pr_1 = label_name_to_pr_curve_1[label]
        precisions_1 = pr_1[0]
        recalls_1 = pr_1[1]

        precision_2 = label_name_to_precision_2[label]
        recall_2 = label_name_to_recall_2[label]

        ax.plot(recalls_1, precisions_1, color=vl.NICE_BLUE, lw=8)
        ax.plot([recall_2], [precision_2], color=vl.NICE_RED, marker='o', markersize=25)
        ax.set_title(label)

    out_f = join(out_dir, "label_to_pr_curve_1_pr_point_2.pdf")
    fig.savefig(out_f, format='pdf', bbox_inches='tight', dpi=200)



def draw_collapsed_ontology_w_pr_curves(
        sample_to_labels,
        label_graph, 
        label_to_name,
        label_to_pr_curve_1, 
        label_to_pr_curve_2,
        label_to_avg_precision_1,
        label_to_avg_precision_2,
        out_dir
    ):
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

    label_to_method_1_better = {}
    label_to_diff_avg_prec = {}
    label_to_fig = {}
    for label in label_to_pr_curve_1:
        avg_pr_1 = label_to_avg_precision_1[label]
        avg_pr_2 = label_to_avg_precision_2[label]
        if avg_pr_1 > avg_pr_2:
            label_to_method_1_better[label] = True
        else:
            label_to_method_1_better[label] = False
        label_to_diff_avg_prec[label] = abs(avg_pr_1 - avg_pr_2) 

        pr_1 = label_to_pr_curve_1[label]
        precisions_1 = pr_1[0]
        recalls_1 = pr_1[1]
        pr_2 = label_to_pr_curve_2[label]
        precisions_2 = pr_2[0]
        recalls_2 = pr_2[1]
        fig, axarr = plt.subplots(
            1,
            1,
            figsize=(3.0, 3.0),
            squeeze=False
        )
        axarr[0][0].plot(recalls_1, precisions_1, color=vl.MAROON, lw=8)
        axarr[0][0].plot(recalls_2, precisions_2, color=vl.SHRILL_BLUE, lw=8)
        title = "%s, (%d)" % (label_to_name[label], len(label_to_samples[label]))

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
        axarr[0][0].set_title(title)
        axarr[0][0].set_xlim(0.0, 1.0)
        axarr[0][0].set_ylim(0.0, 1.0)
        #plt.xticks([], [])
        #plt.yticks([], [])
        out_f = join(tmp_dir, "%s.png" % label_to_name[label].replace(' ', '_').replace('/', '_'))
        fig.savefig(
            out_f, 
            format='png', 
            bbox_inches='tight', 
            dpi=100,
            transparent=True
        )
        label_to_fig[label] = out_f
    result_dot_str = _diff_dot(
        source_to_targets,
        label_to_name, 
        label_to_num_samples, 
        label_to_fig,
        label_to_method_1_better,
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

def make_color_tuple( color ):
    """
    turn something like "#000000" into 0,0,0
    or "#FFFFFF into "255,255,255"
    """
    R = color[1:3]
    G = color[3:5]
    B = color[5:7]

    R = int(R, 16)
    G = int(G, 16)
    B = int(B, 16)

    return R,G,B

def interpolate_tuple( startcolor, goalcolor, steps ):
    """
    Take two RGB color sets and mix them over a specified number of steps.  Return the list
    """
    # white

    R = startcolor[0]
    G = startcolor[1]
    B = startcolor[2]

    targetR = goalcolor[0]
    targetG = goalcolor[1]
    targetB = goalcolor[2]

    DiffR = targetR - R
    DiffG = targetG - G
    DiffB = targetB - B

    buffer = []

    for i in range(0, steps +1):
        iR = R + (DiffR * i / steps)
        iG = G + (DiffG * i / steps)
        iB = B + (DiffB * i / steps)

        hR = string.replace(hex(iR), "0x", "")
        hG = string.replace(hex(iG), "0x", "")
        hB = string.replace(hex(iB), "0x", "")

        if len(hR) == 1:
            hR = "0" + hR
        if len(hB) == 1:
            hB = "0" + hB

        if len(hG) == 1:
            hG = "0" + hG

        color = string.upper("#"+hR+hG+hB)
        buffer.append(color)

    return buffer

def interpolate( startcolor, goalcolor, steps ):
    """
    wrapper for interpolate_tuple that accepts colors as html ("#CCCCC" and such)
    """
    start_tuple = make_color_tuple(startcolor)
    goal_tuple = make_color_tuple(goalcolor)

    return interpolate_tuple(start_tuple, goal_tuple, steps)




def _diff_dot(
        source_to_targets, 
        node_to_label, 
        node_to_num_samples, 
        node_to_image, 
        node_to_is_red, 
        node_to_color_intensity 
    ):
    max_samples = float(math.log(max(node_to_num_samples.values())+1.0))
    max_color_intensity = float(max(node_to_color_intensity.values()))
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())

    reds = interpolate(vl.WHITE, vl.NICE_RED, 100)
    blues = interpolate(vl.WHITE, vl.NICE_BLUE, 100)
    
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        if node not in node_to_image:
            continue
        intensity = (node_to_color_intensity[node]/max_color_intensity)#*255
        red_value = reds[int(intensity * 100.0)]
        blue_value = blues[int(intensity * 100.0)]
        #red_value = '#%02x%02x%02x' % (255, 255-intensity, 255-intensity)
        #blue_value = '#%02x%02x%02x' % (255-intensity, 255-intensity, 255)
        if node_to_is_red[node]:
            color_value = red_value
        else:
            color_value = blue_value
        pen_width = (math.log(node_to_num_samples[node]+1.0)/max_samples)*15
        g += '"%s" [style=filled,  fillcolor="%s", image="%s", penwidth=%d, label=""]\n' % (
            node_to_label[node],
            color_value,
            node_to_image[node],
            pen_width
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















########### Functions for making plots

def plot_comparisons(
        label_to_per_sample_recall_1,
        label_to_per_sample_recall_2,
        label_to_per_sample_prec_1,
        label_to_per_sample_prec_2,
        label_to_per_study_recall_1,
        label_to_per_study_recall_2,
        label_to_per_study_prec_1,
        label_to_per_study_prec_2,
        label_to_per_sample_f1_1,
        label_to_per_sample_f1_2,
        method_1_name,
        method_2_name,
        out_dir
    ):

    method_1_name = method_1_name.replace("_", " ")
    method_2_name = method_2_name.replace("_", " ")

    labels_order = list(label_to_per_sample_recall_1.keys())

    da_per_sample_recall = [
        (
            label_to_per_sample_recall_1[label],
            label_to_per_sample_recall_2[label]
        )
        for label in labels_order
    ]
    df_per_sample_recall = pandas.DataFrame(
        data=da_per_sample_recall,
        columns=[method_1_name, method_2_name]
    )

    da_per_sample_prec = [
        (
            label_to_per_sample_prec_1[label],
            label_to_per_sample_prec_2[label]
        )
        for label in labels_order
    ]
    df_per_sample_prec = pandas.DataFrame(
        data=da_per_sample_prec,
        columns=[method_1_name, method_2_name]
    )

    da_per_study_recall = [
        (
            label_to_per_study_recall_1[label],
            label_to_per_study_recall_2[label]
        )
        for label in labels_order
    ]
    df_per_study_recall = pandas.DataFrame(
        data=da_per_study_recall,
        columns=[method_1_name, method_2_name]
    )

    da_per_study_prec = [
        (
            label_to_per_study_prec_1[label],
            label_to_per_study_prec_2[label]
        )
        for label in labels_order
    ]
    df_per_study_prec = pandas.DataFrame(
        data=da_per_study_prec,
        columns=[method_1_name, method_2_name]
    )

    da_per_sample_f1 = [
        (
            label_to_per_sample_f1_1[label],
            label_to_per_sample_f1_2[label]
        )
        for label in labels_order
    ]
    df_per_sample_f1 = pandas.DataFrame(
        data=da_per_sample_f1,
        columns=[method_1_name, method_2_name]
    )
    
    sns.set_style("dark")
    ax = sns.plt.axes()
    fig, axarr = plt.subplots(
        2,
        3,
        sharey=True,
        figsize=(7.5, 5)
    )

    vis_lib.performance_comparison_scatterplot(
        df_per_sample_recall, 
        method_1_name, 
        method_2_name,
        title="Recall",
        ax=axarr[0][0],
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0)
    )

    vis_lib.performance_comparison_scatterplot(
        df_per_sample_prec, 
        method_1_name, 
        method_2_name,
        title="Precision",
        ax=axarr[0][1],
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0)
    )

    vis_lib.performance_comparison_scatterplot(
        df_per_study_recall, 
        method_1_name, 
        method_2_name,
        title="Study-weighted recall",
        ax=axarr[1][0],
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0)
    )

    vis_lib.performance_comparison_scatterplot(
        df_per_study_prec,  
        method_1_name,
        method_2_name,
        title="Study-weighted precision",
        ax=axarr[1][1],
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0)
    )

    vis_lib.performance_comparison_scatterplot(
        df_per_sample_f1,
        method_1_name,
        method_2_name,
        title="F-1 score",
        ax=axarr[0][2],
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0)
    )


    out_f = join(out_dir, "compare.%s_VS_%s.pdf" % (method_1_name.replace(" ", "_"), method_2_name.replace(" ", "_")))
    sns.plt.tight_layout()
#    fig.savefig(out_f, format='pdf', dpi=1000)
    fig.savefig(out_f, format='pdf', dpi=1000, bbox_inches='tight')
    

#TODO Remove many of these arguments
def plot_per_label_recall_stripplots( 
        label_to_per_sample_recall_1,
        label_to_per_sample_recall_2,
        label_to_per_sample_prec_1,
        label_to_per_sample_prec_2,
        label_to_per_study_recall_1,
        label_to_per_study_recall_2,
        label_to_per_study_prec_1,
        label_to_per_study_prec_2,
        method_1_name,
        method_2_name,
        plot_labels,
        out_dir
    ):

    method_1_name = method_1_name.replace("_", " ")
    method_2_name = method_2_name.replace("_", " ")

    labels_order = list(label_to_per_sample_recall_1.keys())


    da = [
        (
            label_name(label),
            label_to_per_sample_recall_2[label],
            method_2_name
        )
        for label in labels_order
        if label in plot_labels
    ]
    name_to_per_sample_recall_1 = {
        label_name(k):v 
        for k,v in label_to_per_sample_recall_1.iteritems()
    }
    da = sorted(da, key=lambda x: abs(x[1] - name_to_per_sample_recall_1[x[0]]), reverse=True)
    da += [
        (
            label_name(label),
            label_to_per_sample_recall_1[label],
            method_1_name
        )
        for label in labels_order
        if label in plot_labels
    ]

    df = pandas.DataFrame(
        data=da,
        columns=["Label", "Recall", "Method"]
    )

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(5, 0.12*len(label_to_per_sample_recall_1))
    )

    vis_lib.mult_var_strip_plot(
        df,
        "Recall",
        "Label",
        "Method",
        ax=ax
    )

    out_f = join(out_dir, "compare.%s_VS_%s.recall_stripplots.pdf" % (method_1_name.replace(" ", "_"), method_2_name.replace(" ", "_")))
    fig.savefig(out_f, format='pdf', dpi=1000, bbox_inches='tight')

def plot_per_label_precision_stripplots(
        label_to_per_sample_recall_1,
        label_to_per_sample_recall_2,
        label_to_per_sample_prec_1,
        label_to_per_sample_prec_2,
        label_to_per_study_recall_1,
        label_to_per_study_recall_2,
        label_to_per_study_prec_1,
        label_to_per_study_prec_2,
        method_1_name,
        method_2_name,
        plot_labels,
        out_dir
    ):

    method_1_name = method_1_name.replace("_", " ")
    method_2_name = method_2_name.replace("_", " ")

    labels_order = list(label_to_per_sample_prec_1.keys())


    da = [
        (
            label_name(label),
            label_to_per_sample_prec_2[label],
            method_2_name
        )
        for label in labels_order
        if label in plot_labels
    ]
    name_to_per_sample_precision_1 = {
        label_name(k): v 
        for k,v in label_to_per_sample_prec_1.iteritems()
    }
    da = sorted(da, key=lambda x: abs(x[1] - name_to_per_sample_precision_1[x[0]]), reverse=True)
    da += [
        (
            label_name(label),
            label_to_per_sample_prec_1[label],
            method_1_name
        )
        for label in labels_order
        if label in plot_labels
    ]

    df = pandas.DataFrame(
        data=da,
        columns=["Label", "Recall", "Method"]
    )

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(5, 0.12*len(label_to_per_sample_prec_1))
    )

    vis_lib.mult_var_strip_plot(
        df,
        "Recall",
        "Label",
        "Method",
        ax=ax
    )

    out_f = join(out_dir, "compare.%s_VS_%s.precision_stripplots.pdf" % (method_1_name.replace(" ", "_"), method_2_name.replace(" ", "_")))
    fig.savefig(out_f, format='pdf', dpi=1000, bbox_inches='tight')



if __name__ == "__main__":
    main()


