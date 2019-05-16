import matplotlib as mpl
mpl.use('Agg')
import os
from os.path import join
from optparse import OptionParser
import json
import sys
import seaborn as sns
from matplotlib import pyplot as plt
import pandas
import math
import scipy
from scipy.stats import wilcoxon

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import load_boiler_plate_data_for_exp_list as lbpdfel
import project_data_on_ontology as pdoo
import vis_lib as vl

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write output")
    (options, args) = parser.parse_args()

    analysis_output_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    out_dir = options.out_dir

    r = lbpdfel.load_everything_not_data(env_dir, exp_list_name)
    og = r[0]
    label_graph = r[1]
    label_to_name = r[2]
    the_exps = r[3]
    exp_to_labels = r[4]
    exp_to_terms = r[5]
    exp_to_tags = r[6]
    exp_to_study = r[7]
    study_to_exps = r[8]
    exp_to_ms_labels = r[9]

    with open(analysis_output_f, 'r') as f:
        analysis_output = json.load(f)

    homo_label_to_hold_out_recalls = analysis_output["homogeneous_recalls"]
    hetero_label_to_hold_out_recalls = analysis_output["heterogeneous_recalls"]
    label_to_studies = analysis_output["held_out_studies"]    

    homo_label_to_avg_precs = analysis_output["homogeneous_avg_precs"]
    hetero_label_to_avg_precs = analysis_output["heterogeneous_avg_precs"]

    homo_label_to_aucs = analysis_output["homogeneous_aucs"]
    hetero_label_to_aucs = analysis_output["heterogeneous_aucs"]

    label_to_frac_hetero_greater = {}
    label_to_frac_homo_greater = {}
    da = []
    mean_ap_differences_da = []
    for label, homo_list_of_study_to_recalls in homo_label_to_hold_out_recalls.iteritems():
        hetero_list_of_recalls = hetero_label_to_hold_out_recalls[label]
        studies = label_to_studies[label]

        homo_list_of_study_to_avg_precs = homo_label_to_avg_precs[label]
        hetero_avg_precs = hetero_label_to_avg_precs[label]

        homo_list_of_study_to_aucs = homo_label_to_aucs[label]
        hetero_aucs = hetero_label_to_aucs[label]

        ## We don't care to evaluate accuracies on held
        ## out studies for which neither classifier was able
        ## to make a single accurate prediction
        #if max(homo_list_of_study_to_recalls + hetero_accs_list) == 0:
        #    continue 

        n_hetero_greater = 0
        n_homo_greater = 0
        n_total = 0
        held_out_i = 0
        zipped_data = zip(
            studies, 
            homo_list_of_study_to_recalls, 
            hetero_list_of_recalls, 
            homo_list_of_study_to_avg_precs, 
            hetero_avg_precs, 
            homo_list_of_study_to_aucs, 
            hetero_aucs
        )
        for z in zipped_data:
            held_out_study = z[0]
            homo_study_to_recalls = z[1]
            hetero_recalls = z[2]
            homo_study_to_aps = z[3] 
            hetero_aps = z[4]
            homo_study_to_aucs = z[5] # TODO
            hetero_auc = z[6]

            # Iterate through the list of dictionaries that map
            # a study to a 
            study_to_mean_ap = {}
            homo_all_aps = []

            for study, aps in homo_study_to_aps.iteritems():
                mean_ap = sum(aps) / len(aps)
                study_to_mean_ap[study] = mean_ap
                homo_all_aps += aps
            
            mean_homo_ap = sum(homo_all_aps) / len(homo_all_aps)
            mean_hetero_ap = sum(hetero_aps) / len(hetero_aps)
            mean_ap_differences_da.append((
                label_to_name[pdoo.convert_node_from_str(label)],
                study,
                mean_homo_ap,
                mean_hetero_ap
            )) 

            max_homo_ap = max(study_to_mean_ap.values())
            if mean_hetero_ap > max_homo_ap:
                n_hetero_greater += 1
            elif max_homo_ap > mean_hetero_ap:
                n_homo_greater += 1
            n_total += 1
           
            for study, mean_ap in study_to_mean_ap.iteritems():
                da.append((
                    label_to_name[pdoo.convert_node_from_str(label)],
                    'Homogeneous',
                    study,
                    mean_ap,
                ))
            da.append((
                label_to_name[pdoo.convert_node_from_str(label)],
                'Heterogeneous',
                held_out_study,
                mean_hetero_ap
            ))
            held_out_i += 1  
        
        label_to_frac_hetero_greater[label] = (n_hetero_greater, n_total)
        label_to_frac_homo_greater[label] = (n_homo_greater, n_total)
    
    mean_ap_differences_df = pandas.DataFrame(
        data=mean_ap_differences_da,
        columns = [
            'Cell type',
            'Held-out study',
            'Homogeneous\nmean avg. prec.',
            'Heterogeneous\nmean avg. prec.'
        ]
    )

    df = pandas.DataFrame(
        data=da, 
        columns=[
            'Cell type', 
            'Training-type', 
            'Held-out study', 
            'Mean avg. prec'
        ]
    )
    all_labels = sorted(set(label_to_frac_hetero_greater.keys()) - set(["CL:2000001"]))
    n_cols = 5
    n_rows = int(math.ceil(len(label_to_frac_hetero_greater) / float(n_cols)))
    print "Computed %d rows" % n_rows 
    fig, axarr = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=False,
        figsize=(n_cols*4, (n_rows*3)),
        squeeze=False
    )
    sns.set_palette('colorblind')
    label_i = 0
    del_coords = []
    for row_i, ax_row in enumerate(axarr):
        for col_i, ax in enumerate(ax_row):
            if label_i > len(all_labels)-1:
                del_coords.append((row_i, col_i))
                continue
            if (row_i == n_rows-3 or row_i == n_rows - 2) and col_i == n_cols-1:
                del_coords.append((row_i, col_i))
                continue
            label = all_labels[label_i]
            label_name = label_to_name[pdoo.convert_node_from_str(label)]
            curr_df = df.loc[df['Cell type'] == label_name]
            print curr_df
            if label_name == "peripheral blood mononuclear cell":
                continue
            else:
                sns.stripplot(
                    data=curr_df, 
                    x='Mean avg. prec', 
                    y='Held-out study', 
                    hue='Training-type', 
                    dodge=False, 
                    ax=ax, 
                    size=14, 
                    alpha=0.6, 
                    jitter=False
                )
            for ymaj in ax.yaxis.get_majorticklocs():
                ax.axhline(
                    y=ymaj,
                    ls='-',
                    color='grey',
                    alpha=0.5,
                    lw=0.6
                )
            ax.set_title(label_name, fontsize=16)
            if label_i > 0:
                ax.get_legend().set_visible(False)
            ax.set_xlim((0.0, 1.0))
            if col_i > 0:
                ax.set_ylabel("")
            label_i += 1
    for row, col in del_coords:
        fig.delaxes(axarr[row][col])
    out_f = join(out_dir, "compare_recalls_homo_vs_hetero")
    plt.tight_layout()
    fig.savefig("%s.pdf" % out_f, format='pdf', dpi=1000, bbox_inches='tight')
    fig.savefig("%s.eps" % out_f, format='eps', dpi=100, bbox_inches='tight')
    fig.savefig("%s.png" % out_f, format='png', dpi=100, bbox_inches='tight')


    fig, axarr = plt.subplots(
        1,
        1,
        sharex=True,
        sharey=False,
        figsize=(4, 6),
        squeeze=False
    )
    curr_df = df.loc[df['Cell type'] == "peripheral blood mononuclear cell"]
    ax = axarr[0][0]
    sns.stripplot(
        data=curr_df,
        x='Mean avg. prec',
        y='Held-out study',
        hue='Training-type',
        dodge=False,
        ax=ax,
        size=6,
        alpha=0.6,
        jitter=False
    )
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(
            y=ymaj,
            ls='-',
            color='grey',
            alpha=0.5,
            lw=0.6
        )
    ax.set_xlim((0.0, 1.0))
    ax.set_title("peripheral blood mononuclear cell", fontsize=16)
    ax.get_legend().set_visible(False)
    ax.set_ylabel("")
    ax.get_legend().set_visible(False)
    out_f = join(out_dir, "pbmc_compare_recalls_homo_vs_hetero")
    plt.tight_layout()
    fig.savefig("%s.pdf" % out_f, format='pdf', dpi=1000, bbox_inches='tight')
    fig.savefig("%s.eps" % out_f, format='eps', dpi=100, bbox_inches='tight')
    fig.savefig("%s.png" % out_f, format='png', dpi=100, bbox_inches='tight')

    latex_str = """
        \\begin{table}[h!]\caption{}
        \\begin{tabular}{ccc}
        \\hline 
            Cell type  & Het. $>$ Max hom. & Max hom. $>$ Het.\\\\ \\hline
    """
    for label in label_to_frac_hetero_greater:
        frac_hetero = label_to_frac_hetero_greater[label]
        frac_homo = label_to_frac_homo_greater[label]

        print label_to_name[pdoo.convert_node_from_str(label)]
        print "Hetero. $>$ Max homo.:\t%d/%d = %f" % (frac_hetero[0], frac_hetero[1], (float(frac_hetero[0])/frac_hetero[1])) 
        print "Max homo. $>$ Hetero.:\t%d/%d = %f" % (frac_homo[0], frac_homo[1], (float(frac_homo[0])/frac_homo[1]))
        print

        latex_str += "%s & " % label_to_name[pdoo.convert_node_from_str(label)]
        if frac_hetero[0] > 0:
            latex_str += "%d/%d & " % (
                frac_hetero[0],
                frac_hetero[1]
            )
        else:
            latex_str += "- & "

        if frac_homo[0] > 0:
            latex_str += "%d/%d & " % (
                frac_homo[0],
                frac_homo[1]
            )
        else:
            latex_str += "- & "
        latex_str += "\\\\ \n"
    latex_str += """
        \\hline
        \\end{tabular}
        \\end{table}
    """
    print latex_str

    ###############################################################################################

    fig, axarr = plt.subplots(
        1,
        1,
        #figsize=(8, 5),
        figsize=(3.5,3.5),
        squeeze=False
    )
    sns.set_palette('husl')
    ax = axarr[0][0]
    sns.scatterplot(
        data=mean_ap_differences_df, 
        x='Homogeneous\nmean avg. prec.', 
        y='Heterogeneous\nmean avg. prec.',
        #hue='Cell type',
        ax=ax,
        s=40
    ) 
    max_lim = max([ax.get_ylim()[1], ax.get_xlim()[1]])
    print max_lim
    ax.plot([0, max_lim], [0, max_lim], 'k-')
    #ax.get_legend().remove()
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel('Homogeneous mean avg. precision', fontsize=12)
    ax.set_ylabel('Heterogeneous mean avg. precision', fontsize=12)
    out_f = join(out_dir, "compare_avg_precs_homo_heter")
    plt.tight_layout()
    fig.savefig("%s.pdf" % out_f, format='pdf', dpi=1000, bbox_inches='tight') 
    fig.savefig("%s.eps" % out_f, format='eps', dpi=1000, bbox_inches='tight') 


if __name__ == "__main__":
    main()
