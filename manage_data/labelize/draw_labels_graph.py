import sys
import os
from os.path import join

from optparse import OptionParser
import json
import collections
from collections import defaultdict
import numpy as np
import math
import subprocess

sys.path.append("..")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")

from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import project_data_on_ontology as pdoo
import load_experiment_lists as lel
import the_ontology

ONTOLOGY = the_ontology.the_ontology()


def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    env_root = args[0]
    exp_list_name = args[1]

    data_info_f = join(
        env_root, 
        "data", 
        "data_set_metadata.json"
    )
    data_partition_f = join(
        env_root, 
        "data", 
        "data_partition.json"
    )
    with open(data_info_f, 'r') as f:
        exp_to_info = json.load(f)

    # load data
    the_exps = lel.union_of_experiment_lists(
        env_root,
        [exp_list_name]
    )

    out_dir = join(
        env_root,
        "data",
        "experiment_lists",
        exp_list_name,
        "summary_figures"
    )
    
    # Draw the collapsed ontology labels
    labels_f = join(
        env_root, 
        "data", 
        "experiment_lists", 
        exp_list_name, 
        "labelling_collapsed_dag.json"
    )
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
    output_prefix = "collapsed_ontology"
    draw_labels_graphs(
        the_exps,
        exp_to_info, 
        labels_data, 
        output_prefix, 
        out_dir
    )

    # Draw the full ontology labels
    labels_f = join(
        env_root,
        "data",
        "experiment_lists",
        exp_list_name,
        "labelling.json"
    )
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
    output_prefix = "full_ontology"
    draw_labels_graphs(
        the_exps,
        exp_to_info, 
        labels_data, 
        output_prefix, 
        out_dir
    )

def draw_labels_graphs(
        the_exps,
        exp_to_info, 
        labels_data, 
        output_prefix, 
        out_dir
    ):

    # load labellings 
    jsonable_graph = labels_data['label_graph']
    jsonable_exp_to_labels = labels_data['labelling']
    label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    all_exp_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_exp_to_labels)

    the_exp_to_labels = {
        exp: labels 
        for exp, labels in all_exp_to_labels.iteritems() 
        if exp in the_exps
    }
    the_sample_to_labels = {
        exp_to_info[exp]['sample_accession']: labels
        for exp, labels in the_exp_to_labels.iteritems()
    }
    sample_to_study = {
        exp_to_info[exp]['sample_accession']: exp_to_info[exp]['study_accession']
        for exp in the_exps
    }
 
    source_to_targets = label_graph.source_to_targets 
    target_to_sources = label_graph.target_to_sources

    label_to_samples = defaultdict(lambda: set())
    ms_label_to_samples = defaultdict(lambda: set())
    label_to_studies = defaultdict(lambda: set())
    for sample, labels in the_sample_to_labels.iteritems():
        study = sample_to_study[sample]
        ms_labels = label_graph.most_specific_nodes(labels)
        for label in ms_labels:
            ms_label_to_samples[label].add(sample)
        for label in labels:
            label_to_samples[label].add(sample)
            label_to_studies[label].add(study)
    label_to_num_samples = {
        k: len(v) 
        for k,v in 
        label_to_samples.iteritems()
    }
    ms_label_to_num_samples = {
        k: len(v)
        for k,v in
        ms_label_to_samples.iteritems()
    }
    for label in label_graph.get_all_nodes():
        if label not in ms_label_to_num_samples:
            ms_label_to_num_samples[label] = 0 
    label_to_num_studies = {
        k: len(v) 
        for k,v in 
        label_to_studies.iteritems()
    }

    result_dot_str = dot_names(
        source_to_targets, 
        ms_label_to_num_samples, 
        ONTOLOGY
    )
    dot_f = join(
        out_dir,
        "%s.by_name_num_samples_most_specific.dot" % output_prefix 
    )
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    run_cmd(
        "dot -Tpdf %s -o %s" % (
            dot_f, 
            join(
                out_dir,
                "%s.by_name_num_samples.pdf" % output_prefix
            )
        )
    )

    result_dot_str = dot_names(
        source_to_targets, 
        label_to_num_studies, 
        ONTOLOGY
    )
    dot_f = join(
        out_dir,
        "%s.by_name_num_studies.dot" % output_prefix
    )
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    run_cmd(
        "dot -Tpdf %s -o %s" % (
            dot_f,
            join(
                out_dir,
                "%s.by_name_num_studies.pdf" % output_prefix
            )
        )
    )

    result_dot_str = dot_ids(
        source_to_targets, 
        label_to_num_samples, 
        ONTOLOGY
    )
    dot_f = join(
        out_dir,
        "%s.by_id.dot" % output_prefix
    )
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    run_cmd(
        "dot -Tpdf %s -o %s" % (
            dot_f,  
            join(
                out_dir,
                "%s.by_id.pdf" % output_prefix
            )
        )
    )


def dot_ids(source_to_targets, node_to_num_samples, og):
    max_samples = float(math.log(max(node_to_num_samples.values())+1.0))
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        color_value = (1.0-(math.log(node_to_num_samples[node]+1.0)/max_samples))*255
        color_value = '#%02x%02x%02x' % (255, color_value, 255)
        g += '"%s (%d)" [style=filled,  fillcolor="%s"]\n' % (
            "\\n".join([x for x in node]),
            node_to_num_samples[node],
            color_value
        )
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            g += '"%s (%d)" -> "%s (%d)"\n' % (
                "\\n".join([x for x in source]),
                node_to_num_samples[source],
                "\\n".join([x for x in target]),
                node_to_num_samples[target]
            )
    g += "}"
    return g


def dot_names(source_to_targets, node_to_num_samples, og):
    max_samples = float(math.log(max(node_to_num_samples.values())+1.0))
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        color_value = (1.0-(math.log(node_to_num_samples[node]+1.0)/max_samples))*255
        color_value = '#%02x%02x%02x' % (255, color_value, 255)
        g += '"%s (%d)" [style=filled,  fillcolor="%s"]\n' % (
            "\\n".join([og.id_to_term[x].name for x in node]),
            node_to_num_samples[node],
            color_value
        )
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            g += '"%s (%d)" -> "%s (%d)"\n' % (
                "\\n".join([og.id_to_term[x].name for x in source]),
                node_to_num_samples[source],
                "\\n".join([og.id_to_term[x].name for x in target]),
                node_to_num_samples[target]
            )
    g += "}"
    return g
    

def run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)


if __name__ == "__main__":
    main()

