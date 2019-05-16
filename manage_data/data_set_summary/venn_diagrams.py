from optparse import OptionParser
import sys
from collections import defaultdict
import json
import os
from os.path import join
import random

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/vis_lib")

import load_boiler_plate_data_for_exp_list as lbpdfel
#import vis_lib
import nvenn

PLOT_TERMS = [
#    "CL:0000988",   # hematopoietic cell
    "CL:0002321",   # embryonic cell
    "CL:0011115",   # precursor cell
#    "CL:0002319",   # neural cell
    "CL:0002371",   # somatic cell
    "CL:0000039",   # germ line cell
#    "CL:0000066",   # epithelial cell
#    "CL:0002320",   # connective tissue cell
#    "other"
]

PLOTS = [
    {
        "condition_on": None,
        "terms": [
            "CL:0002321",   # embryonic cell
            "CL:0011115",   # precursor cell
            "CL:0002371",   # somatic cell
            "CL:0000039"    # germ line cell
        ],
        "term_names": [
            "embryonic cell",
            "precursor cell",
            "somatic cell",
            "germ line cell"
        ],
        "out_f": "major_divisions.svg"
    },
    {
        "condition_on": "CL:0002371",   # somatic cell,
        "terms": [
            "CL:0000988",
            "CL:0002319",
            "CL:0000066",
            "CL:0002320",
            "CL:0011115"
        ],
        "term_names": [
            'hematopoietic cell',
            'neural cell',
            'epithelial cell',
            'connective tissue cell',
            'precursor cell'
        ],
        "out_f": "somatic_divisions.svg"
    },
    {
        "condition_on": "CL:0000988",   # hematopoietic cell,
        "terms": [
            "CL:0000738",
            "CL:0000763",
            "CL:0000542",
            "CL:0008001",
            "CL:1001610"
        ],
        "term_names": [
            'leukocyte',
            'myeloid cell',
            'lymphocyte',
            'hematopoietic precursor cell',
            'bone marrow hematopoietic cell'
        ],
        "out_f": "hematopoietic_divisions.svg"
    },
    {
        "condition_on": "CL:0000039",   # germ line cell
        "terms": [
            "CL:0000015",   # male germ cell
            "CL:0000021",   # female germ cell
            "CL:0000014",   # germ line stem cell
            "CL:0000670"    # primordial germ cell
        ],
        "term_names": [
            "male germ cell",
            "female germ cell",
            "germ line stem cell",
            "primordial germ cell"
        ],
        "out_f": "germ_divisions.svg"
    }
]

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write the output")
    (options, args) = parser.parse_args()
    
    env_dir = args[0]
    exp_list_name = args[1]
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


    term_to_exps = defaultdict(lambda: set())
    no_plot_terms_exps = set()
    for exp, terms in exp_to_terms.iteritems():
        for term in terms:
            term_to_exps[term].add(exp)


    print "Oh no! The following terms are annotated with both germ and somatic cell:"
    print term_to_exps["CL:0000039"] & term_to_exps["CL:0002371"]

    for plot in PLOTS:
        cond_term = plot['condition_on']
        if cond_term is None:
            cand_exps = set(the_exps)
        else:
            cand_exps = set(term_to_exps[cond_term])
        plot_terms = set(plot['terms'])
        exps_no_plot_terms = set()
        for exp in cand_exps:
            exp_plot_terms = plot_terms & set(exp_to_terms[exp])
            if len(exp_plot_terms) == 0:
                exps_no_plot_terms.add(exp)
        #sets = [exps_no_plot_terms]
        #set_names = ["other"]
        if len(exps_no_plot_terms) > 0:
            set_name_to_set = {"other": exps_no_plot_terms}
        else:
            set_name_to_set = {}
        for plot_term_name, plot_term in zip(plot['term_names'], plot['terms']):
            #sets.append(
            #    set(term_to_exps[plot_term]) & cand_exps
            #)
            #set_names.append(plot_term_name)
            set_name_to_set[plot_term_name] = set(term_to_exps[plot_term]) & cand_exps

        sets = []
        set_names = []
        set_name_set_tuples = [x for x in set_name_to_set.iteritems()]
        random.shuffle(set_name_set_tuples)
        for set_name, sett in set_name_set_tuples:
        # for set_name, sett in sorted(set_name_to_set.iteritems(), key=lambda x: x[0]):
            set_names.append(set_name)
            sets.append(sett)

        if len(sets) <= 4:
            #n_cycles = 7
            n_cycles = 2
        else:
            n_cycles = 15

        cwd = os.getcwd()
        nvenn.create_nvenn(
            sets,
            set_names,
            join(out_dir, plot['out_f'].replace("svg", "tmp")),
            join(out_dir, plot['out_f']),
            #n_cycles=2*len(sets),
            n_cycles=n_cycles,
            delete_tmp_dir=False
        ) 
            
 
if __name__ == "__main__":
    main()
