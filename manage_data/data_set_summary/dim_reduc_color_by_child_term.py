###########################################################################
#   This script generate summary stats and plots in regards to the library
#   selection tags. Originally this script was developed to get a sesne of
#   the dataset to be used for training an RNA-seq library prep strategy
#   classifier.
###########################################################################

import matplotlib as mpl
mpl.use('Agg')

from optparse import OptionParser
from collections import defaultdict
import json
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
import os
from os.path import join
import subprocess
import math

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")

import labelizers
import featurizers
import vis_lib
import the_ontology
import project_data_on_ontology as pdoo
import graph_lib
from graph_lib import graph
import kallisto_quantified_data_manager_hdf5 as kqdm
from sklearn import decomposition
from sklearn import manifold


def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Output file")
    (options, args) = parser.parse_args()

    env_dir = args[0]
    exp_list_name = args[1]
    out_dir = options.out_dir
   
    exp_info_f = join(env_dir, "data", "data_set_metadata.json")
    exp_list_f = join(env_dir, "data", "experiment_lists", exp_list_name, "experiment_list.json")
    labels_f = join(env_dir, "data", "experiment_lists", exp_list_name, "labelling.json")
     
#    exp_info_f = "/tier2/deweylab/mnbernstein/phenotyping_environments/cell_type.metasra_1-4/data/data_set_metadata.json"
#    exp_list_f = "/tier2/deweylab/mnbernstein/phenotyping_environments/cell_type.metasra_1-4/data/experiment_lists/all_experiments_with_data/experiment_list.json"
#    labels_f = "/tier2/deweylab/mnbernstein/phenotyping_environments/cell_type.metasra_1-4/data/experiment_lists/all_experiments_with_data/labelling.json"

    og = the_ontology.the_ontology()

    with open(exp_info_f, 'r') as f:
        exp_to_info = json.load(f)
    with open(exp_list_f, 'r') as f:
        the_exps = json.load(f)['experiments']
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)

    exp_to_sample = {
        exp: exp_to_info[exp]['sample_accession']
        for exp in exp_to_info
    }
    exp_to_study = {
        exp: exp_to_info[exp]['study_accession']
        for exp in exp_to_info
    }    
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.iteritems():
        study_to_exps[study].add(exp)

    # load labellings 
    jsonable_graph = labels_data['label_graph']
    jsonable_exp_to_labels = labels_data['labelling']
    label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    exp_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_exp_to_labels)
    label_to_name = {
        x: pdoo.convert_node_to_name(x, og)
        for x in label_graph.get_all_nodes()
    }

    assert frozenset(exp_to_labels.keys()) == frozenset(the_exps) 

    # Map each label to an assigned color
    label_to_color = {}
    for label in label_graph.get_all_nodes():
        color_index = 0

        child_colors = set([
            label_to_color[child]
            for child in label_graph.source_to_targets[label]
            if child in label_to_color
        ])
        for child in label_graph.source_to_targets[label]:
            if not child in label_to_color:
                col = vis_lib.MANY_COLORS[color_index]
                while col in child_colors:
                    color_index += 1
                    col = vis_lib.MANY_COLORS[color_index]
                label_to_color[child] = vis_lib.MANY_COLORS[color_index]
            color_index += 1              
    label_to_new_color = {} 
    for label, col in label_to_color.iteritems():
        sibling_cols = set()
        siblings = set()
        for parent in label_graph.target_to_sources[label]:
            siblings.update(
                label_graph.source_to_targets[parent]
            )
        siblings -= set([label])
        sibling_cols = set([
            label_to_new_color[sib]
            for sib in siblings
            if sib in label_to_new_color
        ])
        sibling_cols.update(set([
            label_to_color[sib]
            for sib in siblings
            if sib not in label_to_new_color 
            and sib in label_to_color
        ]))
        new_col = col
        color_index = 0
        while new_col in sibling_cols:
            color_index += 1
            new_col = vis_lib.MANY_COLORS[color_index]
        label_to_new_color[label] = new_col
    assert frozenset(label_to_color.keys()) == frozenset(label_to_new_color.keys()) 
    label_to_color = label_to_new_color

    # map each term to its label
    term_to_label = {}
    for label in label_graph.get_all_nodes():
        for term in label:
            term_to_label[term] = label

    # map each label to its set of experiments
    label_to_exps = defaultdict(lambda: set())
    for exp, labels in exp_to_labels.iteritems():
        for label in labels:
            label_to_exps[label].add(exp)

    #data_matrix = np.random.rand(len(the_exps), 100)
    the_samples, the_exps, data_matrix, gene_names = featurizers.log_kallisto_gene_counts(
        [
            exp_to_sample[exp]
            for exp in the_exps
        ]
    )

    sns.set_style("dark")

    draw_collapsed_ontology_w_pr_curves(
        exp_to_labels,
        label_to_exps,
        label_graph,
        label_to_name,
        label_to_color,
        exp_to_study, 
        the_exps,
        data_matrix,
        out_dir
    )



def draw_collapsed_ontology_w_pr_curves(
        exp_to_labels,
        label_to_exps,
        label_graph,
        label_to_name,
        label_to_color,
        exp_to_study,
        the_exps,
        data_matrix,
        out_dir
    ):

    exp_to_index = {
        exp: exp_i
        for exp_i, exp in enumerate(the_exps)
    }
    tmp_dir = join(out_dir, "tmp_figs_color_by_child")
    _run_cmd("mkdir -p %s" % tmp_dir)

    source_to_targets = label_graph.source_to_targets
    target_to_sources = label_graph.target_to_sources

    # Map each study to a color
    studies = set(exp_to_study.values())
    study_to_color = {
        study: vis_lib.MANY_COLORS[study_i % len(vis_lib.MANY_COLORS)]
        for study_i, study in enumerate(studies)
    }

    label_to_fig = {}
    for label in label_to_exps:
        child_labels = label_graph.source_to_targets[label]

        label_exps = list(label_to_exps[label])
        
        if len(label_exps) < 2:
            continue
        print "There are %d experiments for label %s" % (len(label_exps), label_to_name[label])
        
        label_exp_to_index = {
            exp: exp_i
            for exp_i, exp in enumerate(label_exps) 
        }
        label_data_matrix = [
            data_matrix[exp_to_index[exp]]
            for exp in label_exps
        ]
        studies = set([
            exp_to_study[exp]
            for exp in label_exps
        ])
        print "%d studies in experiments of interest for label %s" % (len(studies), label_to_name[label])

        print "fitting PCA for label %s..." % label_to_name[label]
        model = decomposition.PCA(2)
        model.fit(label_data_matrix)
        pca_matrix = model.transform(label_data_matrix)
        print "done."

        print "fitting MDS for label %s..." % label_to_name[label]
        model = manifold.MDS(2) 
        mds_matrix = model.fit_transform(label_data_matrix)
        print "done."

        #print "fitting t-SNE for label %s..." % label_to_name[label]
        #model = manifold.TSNE(2)
        #tsne_matrix = model.fit_transform(label_data_matrix)
        #print "done."

        fig, axarr = plt.subplots(
            2,
            2,
            figsize=(6.0, 6.0)
        )
       
        exps_no_child_labels = set([
            exp
            for exp in label_exps
            if len(set(child_labels) & set(exp_to_labels[exp])) == 0
        ])
       
        exp_to_num_dots = {
            exp: 0
            for exp in label_exps
        }
        child_labels_order = list(child_labels)
        for exp in set(label_exps) - exps_no_child_labels:
            exp_child_labels = set(child_labels) & set(exp_to_labels[exp])
            dot_count = 0
            for child in child_labels_order:
                if child not in exp_child_labels:
                    continue
                df_pca = pandas.DataFrame(
                    data=[
                        pca_matrix[label_exp_to_index[exp]]
                    ],
                    columns = ["Component 1", "Component 2"]
                )
                sns.regplot(
                    ax=axarr[0][0],
                    x="Component 1",
                    y="Component 2",
                    data=df_pca,
                    scatter=True,
                    logistic=False,
                    fit_reg=False,
                    ci=None,
                    scatter_kws={
                        "color": label_to_color[child],
                        "linewidth":0,
                        "s":80 / math.pow(2.0, dot_count),
                        "alpha":1
                    }
                )
                df_mds = pandas.DataFrame(
                    data=[
                        mds_matrix[label_exp_to_index[exp]]
                    ],
                    columns = ["Component 1", "Component 2"]
                ) 
                sns.regplot(
                    ax=axarr[1][0],
                    x="Component 1",
                    y="Component 2",
                    data=df_mds,
                    scatter=True,
                    logistic=False,
                    fit_reg=False,
                    ci=None,
                    scatter_kws={
                        "color": label_to_color[child],
                        "linewidth":0,
                        "s":80 / math.pow(2.0, dot_count),
                        "alpha":1
                    }
                )
                #df_tsne = pandas.DataFrame(
                #    data=[
                #        tsne_matrix[label_exp_to_index[exp]]
                #    ],
                #    columns = ["Component 1", "Component 2"]
                #)
                #sns.regplot(
                #    ax=axarr[2][0],
                #    x="Component 1",
                #    y="Component 2",
                #    data=df_tsne,
                #    scatter=True,
                #    logistic=False,
                #    fit_reg=False,
                #    ci=None,
                #    scatter_kws={
                #        "color": label_to_color[child],
                #        "linewidth":0,
                #        "s":80 / math.pow(2.0, dot_count),
                #        "alpha":1
                #    }
                #)
                #dot_count += 1

        # Color the experiments that are not marked with a child label
        # as black
        df_pca = pandas.DataFrame(
            data=[
                pca_vec
                for pca_i, pca_vec in enumerate(pca_matrix)
                if label_exps[pca_i] in exps_no_child_labels
            ],
            columns = ["Component 1", "Component 2"]
        )
        sns.regplot(
            ax=axarr[0][0],
            x="Component 1",
            y="Component 2",
            data=df_pca,
            scatter=True,
            logistic=False,
            fit_reg=False,
            ci=None,
            scatter_kws={
                "color": "black",
                "linewidth":0,
                "s":100,
                "alpha":1
            }
        )
        df_mds = pandas.DataFrame(
            data=[
                mds_vec
                for mds_i, mds_vec in enumerate(mds_matrix)
                if label_exps[mds_i] in exps_no_child_labels
            ],
            columns = ["Component 1", "Component 2"]
        )
        sns.regplot(
            ax=axarr[1][0],
            x="Component 1",
            y="Component 2",
            data=df_mds,
            scatter=True,
            logistic=False,
            fit_reg=False,
            ci=None,
            scatter_kws={
                "color": "black",
                "linewidth":0,
                "s":100,
                "alpha":1
            }
        )
        #df_tsne = pandas.DataFrame(
        #    data=[
        #        tsne_vec
        #        for tsne_i, tsne_vec in enumerate(tsne_matrix)
        #        if label_exps[tsne_i] in exps_no_child_labels
        #    ],
        #    columns = ["Component 1", "Component 2"]
        #)
        #sns.regplot(
        #    ax=axarr[2][0],
        #    x="Component 1",
        #    y="Component 2",
        #    data=df_tsne,
        #    scatter=True,
        #    logistic=False,
        #    fit_reg=False,
        #    ci=None,
        #    scatter_kws={
        #        "color": "black",
        #        "linewidth":0,
        #        "s":100,
        #        "alpha":1
        #    }
        #)


        # Create the second plot, but color by study
        print "PCA MATRIX SHAPE: %s" % str(pca_matrix.shape)
        for curr_study in studies:
            df_pca = pandas.DataFrame(
                data=[
                    pca_matrix[exp_i]
                    for exp_i, exp in enumerate(label_exps)
                    if exp_to_study[exp] == curr_study
                ],
                columns = ["Component 1", "Component 2"]
            )
            sns.regplot(
                ax=axarr[0][1],
                x="Component 1",
                y="Component 2",
                data=df_pca,
                scatter=True,
                logistic=False,
                fit_reg=False,
                ci=None,
                scatter_kws={
                    "color": study_to_color[curr_study],
                    "linewidth":0,
                    "s":100,
                    "alpha":1
                }
            )
            df_mds = pandas.DataFrame(
                data=[
                    mds_matrix[exp_i]
                    for exp_i, exp in enumerate(label_exps)
                    if exp_to_study[exp] == curr_study
                ],
                columns = ["Component 1", "Component 2"]
            )
            sns.regplot(
                ax=axarr[1][1],
                x="Component 1",
                y="Component 2",
                data=df_mds,
                scatter=True,
                logistic=False,
                fit_reg=False,
                ci=None,
                scatter_kws={
                    "color": study_to_color[curr_study],
                    "linewidth":0,
                    "s":100,
                    "alpha":1
                }
            )
            #df_tsne = pandas.DataFrame(
            #    data=[
            #        tsne_matrix[exp_i]
            #        for exp_i, exp in enumerate(label_exps)
            #        if exp_to_study[exp] == curr_study
            #    ],
            #    columns = ["Component 1", "Component 2"]
            #)
            #sns.regplot(
            #    ax=axarr[2][1],
            #    x="Component 1",
            #    y="Component 2",
            #    data=df_tsne,
            #    scatter=True,
            #    logistic=False,
            #    fit_reg=False,
            #    ci=None,
            #    scatter_kws={
            #        "color": study_to_color[curr_study],
            #        "linewidth":0,
            #        "s":100,
            #        "alpha":1
            #    }
            #)
        
        plt.tight_layout()
        

        title = label_to_name[label]

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
        #ax.set_title(title)
        fig.suptitle(title)
        for r in range(len(axarr)):
            for c in range(len(axarr[0])):
                axarr[r][c].set_xlabel("")
                axarr[r][c].set_ylabel("")
                axarr[r][c].set_xticks([])
                axarr[r][c].set_yticks([])
        
        for r in range(len(axarr)):
            y_lim = axarr[r][0].get_ylim()
            x_lim = axarr[r][0].get_xlim() 
            for c in range(len(axarr[0])):
                axarr[r][c].set_xlim(x_lim)
                axarr[r][c].set_ylim(y_lim)

        out_f = join(tmp_dir, "%s.png" % label_to_name[label].replace(' ', '_').replace('/', '_'))
        fig.savefig(
            out_f,
            format='png',
            bbox_inches='tight',
            dpi=100,
            transparent=False
        )
        label_to_fig[label] = out_f
    result_dot_str = _diff_dot(
        source_to_targets,
        label_to_name,
        label_to_fig,
        label_to_color
    )
    dot_f = join(out_dir, "pca_color_by_child_on_graph.dot")
    graph_out_f = join(out_dir, "pca_color_by_child_on_graph.pdf")
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, graph_out_f))
    _run_cmd("rm -r %s" % tmp_dir)

def _diff_dot(
        source_to_targets,
        node_to_label,
        node_to_image,
        node_to_color
    ):
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        if node in node_to_image and node not in node_to_color:
            g += '"%s" [image="%s", label="", penwidth=5]\n' % (
                node_to_label[node],
                node_to_image[node]
            )
        elif node in node_to_image:
            g += '"%s" [image="%s", label="", penwidth=12, color="%s"]\n' % (
                node_to_label[node],
                node_to_image[node],
                node_to_color[node]
            )
        else:
            g += '"%s" [color="%s", penwidth=12]\n' % (
                node_to_label[node], 
                node_to_color[node]
            )
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            g += '"%s" -> "%s"\n' % (
                node_to_label[source],
                node_to_label[target]
            )
    g += "}"
    return g


def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)



if __name__ == "__main__":
    main()
