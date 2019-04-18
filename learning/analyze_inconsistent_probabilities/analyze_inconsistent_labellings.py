###########################################################################################
#
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
import string

sys.path.append("/ua/mnbernstein/projects/vis_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/learning/evaluation")

import project_data_on_ontology as pdoo
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import compute_metrics as cm
import vis_lib as vl
import the_ontology
import graph_lib
from graph_lib import graph

EPSILON = 0.000000000001

def main():
    usage = "usage: %prog <options> <environment dir> <experiment list name> <cross-validation config name>"
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write the output")
    parser.add_option("-l", "--alt_label_graph_f", help="Use this label graph instead of another") # TODO not sure if this will cause bugs...
    (options, args) = parser.parse_args()
    
    env_dir = args[0]
    exp_list_name = args[1]
    results_f = args[2]
    out_dir = options.out_dir

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

    with open(results_f, 'r') as f:
        results = json.load(f)
    exp_to_label_to_conf = results['predictions']

    print "NUMBER OF PREDICTIONS: %d" % len(exp_to_label_to_conf)

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
   
    cm.validate_predictions(
        exp_to_label_to_conf,
        include_labels
    )
 
    label_to_name = {
        x: pdoo.convert_node_to_name(x, og)
        for x in label_graph.get_all_nodes()
    }


    total_n_incons = 0
    total_n_very_incons = 0
    total_edges = 0
    incons_to_count = defaultdict(lambda: 0)
    very_incons_to_count = defaultdict(lambda: 0)
    exp_child_parent_incons = []
    for exp, label_to_conf in exp_to_label_to_conf.iteritems():
        exp_n_incons = 0
        for parent_label, parent_conf in label_to_conf.iteritems():
            for child_label in label_graph.source_to_targets[parent_label]:
                if child_label not in label_to_conf:
                    continue
                child_conf = label_to_conf[child_label]
                if child_conf < 0.01 and parent_conf < 0.01:
                    continue
                if abs(child_conf - parent_conf) > EPSILON and child_conf > parent_conf:
                    exp_child_parent_incons.append((exp, child_label, parent_label, (child_conf-parent_conf)))
                    incons_to_count[(parent_label, child_label)] += 1
                    total_n_incons += 1
                    exp_n_incons += 1
                    if child_conf - parent_conf > 0.5:
                        total_n_very_incons += 1
                        very_incons_to_count[(parent_label, child_label)] += 1
                total_edges += 1
    total_fraction_inconsistent = total_n_incons / float(total_edges)
    total_fraction_very_inconsistent = total_n_very_incons / float(total_edges)

    print "Inconsistent edges:"
    for incons, count in sorted([(k,v) for k,v in incons_to_count.iteritems()], key=lambda x: x[1]):
        parent = incons[0]
        child = incons[1]
        print "%s -> %s : %d" % (label_to_name[parent], label_to_name[child], count)
    print "Very inconsistent edges:"
    for incons, count in sorted([(k,v) for k,v in very_incons_to_count.iteritems()], key=lambda x: x[1]):
        parent = incons[0]
        child = incons[1]
        print "%s -> %s : %d" % (label_to_name[parent], label_to_name[child], count)

    print "Total edges inconsistent %d/%d = %f" % (total_n_incons, total_edges, 
        total_fraction_inconsistent)
    print "Total edges very inconsistent: %d/%d = %f" % (
        total_n_very_incons, total_edges, total_fraction_very_inconsistent)
    print "Avg. very inconsistent per sample: %d/%d=%f" % (
        total_n_very_incons, len(exp_to_label_to_conf), 
        (float(total_n_very_incons)/len(exp_to_label_to_conf)))

    with open(join(out_dir, 'inconsistent_edges_stats.tsv'), 'w') as f:
        f.write("Total edges inconsistent\t%d/%d\t%f\n" % (total_n_incons, total_edges, total_fraction_inconsistent))
        f.write("Total edges very inconsistent\t%d/%d\t%f\n" % (total_n_very_incons, total_edges, total_fraction_very_inconsistent))
        f.write("Avg. very inconsistent per sample\t%d/%d\t%f\n" % (total_n_very_incons, len(exp_to_label_to_conf), (float(total_n_very_incons)/len(exp_to_label_to_conf))))

    exp_child_parent_incons = sorted(exp_child_parent_incons, key=lambda x: x[3])
    inconss = []
    n_less_eq = []
    l_less_than_1 = 0
    n_great_than_1 = 0
    for i, exp_child_parent_icons in enumerate(exp_child_parent_incons):
        incons = exp_child_parent_icons[3]
        #if incons > 0.01:
        inconss.append(incons)
        n_less_eq.append(float(i)/len(exp_child_parent_incons))
        #else:
        #    l_less_than_1 += 1
    #print "Total inconsistent but less than 0.01: %d %f" % (l_less_than_1, (l_less_than_1/float(total_edges)))
    #print "Total inconsistent grater than 0.01 per sample: %d/%d=%f" % (n_great_than_1, len(exp_to_label_to_conf), (float(total_n_very_incons)/len(exp_to_label_to_conf)))

    fig, axarr = plt.subplots( 
        1, 
        1, 
        figsize=(3.0, 3.0), 
        squeeze=False 
    ) 
    axarr[0][0].plot(inconss, n_less_eq, color=vl.NICE_COLORS[1], lw=4)
    axarr[0][0].set_xlabel('Child prob. - Parent prob.')
    axarr[0][0].set_ylabel('Cumulative probability')
    axarr[0][0].set_xlim((0.0, 1.0)) 
    axarr[0][0].set_ylim((0.0, 1.0))
    out_f = join(out_dir, "CDF_inconsistences") 
    fig.savefig( 
        "%s.eps" % out_f, 
        format='eps', 
        bbox_inches='tight', 
        dpi=100, 
        transparent=True 
    )
    fig.savefig(
        "%s.pdf" % out_f,
        format='pdf',
        bbox_inches='tight',
        dpi=100,
        transparent=True
    ) 
    
           

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
        node_to_num_exps,
        node_to_image,
        node_to_color_intensity
    ):
    max_exps = float(math.log(max(node_to_num_exps.values())+1.0))
    max_color_intensity = float(max(node_to_color_intensity.values()))
    print "MAX INTENSITY: %f" % max_color_intensity
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)

    reds = interpolate(vl.WHITE, vl.NICE_RED, 100)
    for node in all_nodes:
        if node not in node_to_image:
            continue
        intensity = (node_to_color_intensity[node]/max_color_intensity)#*255
        #color_value = '#%02x%02x%02x' % (255, 255-intensity, 255-intensity) # red
        color_value = reds[int(intensity * 100.0)]
        #color_value = '#%02x%02x%02x' % (255-intensity, 255-intensity, 255) # blue
        pen_width = (math.log(node_to_num_exps[node]+1.0)/max_exps)*15
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


def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)



























########### Functions for making plots

if __name__ == "__main__":
    main()


