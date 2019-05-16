####################################################################################
#   This creates a list of test experiments to be used to test various programs used
#   in my analyses/experiments.
#
#   This is a subset of the 'untampered' experiments, by which I mean all 
#   experiments that have been treated in such a way that a purposeful change in 
#   gene expression was induced.  This does not include experiments that have been 
#   transfected with a control vector.
####################################################################################

from optparse import OptionParser
import sys
import json
from collections import defaultdict 

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/graph_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import labelizers
import the_ontology
import graph
import kallisto_quantified_data_manager_hdf5 as kqdm

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    exp_info_f = args[0]
    untampered_exps_w_data_list_f = args[1]    
    out_f = options.out_file

    og = the_ontology.the_ontology()

    with open(untampered_exps_w_data_list_f, 'r') as f:
        include_experiments_data = json.load(f)
    with open(exp_info_f, 'r') as f:
        exp_to_info = json.load(f)

    include_experiments = set(include_experiments_data['experiments'])
    parent_exp_list_name = include_experiments_data['list_name']
    assert parent_exp_list_name == "untampered_bulk_primary_cells_with_data"

    exp_to_study = {
        exp: exp_to_info[exp]['study_accession']
        for exp in exp_to_info
    }
    sample_to_exps = defaultdict(lambda: set())
    for exp, info in exp_to_info.iteritems():
        sample = info['sample_accession']
        sample_to_exps[sample].add(exp)
    
    exp_to_ms_cell_types = labelizers.cell_type_labelizer(
        include_experiments, 
        exp_to_info
    )
    ms_cell_type_to_exps = defaultdict(lambda: set())
    for exp, terms in exp_to_ms_cell_types.iteritems():
        for term in terms:
            ms_cell_type_to_exps[term].add(exp)
    ms_cell_type_to_studies = defaultdict(lambda: set())
    for term, exps in ms_cell_type_to_exps.iteritems():
        ms_cell_type_to_studies[term].update([
            exp_to_study[exp]
            for exp in exps
        ])
    study_to_exps = defaultdict(lambda: set())
    for exp in include_experiments:
        study = exp_to_study[exp]
        study_to_exps[study].add(exp) 

    cell_type_to_chosen_studies = {}
    for term, studies in ms_cell_type_to_studies.iteritems():
        if term not in cell_type_to_chosen_studies:
            sorted_studies = sorted(studies)
            chosen_studies = [sorted_studies[0]]
            if len(sorted_studies) > 1:
                chosen_studies.append(sorted_studies[1])
            anc_terms = og.recursive_superterms(term) 
            for anc_term in anc_terms:
                cell_type_to_chosen_studies[anc_term] = chosen_studies    
        elif len(cell_type_to_chosen_studies[term]) >= 2: # We already have studies representing this term
            continue
        else: # There is currently one study for the current cell type and we should attempt to grab one more
            candidate_studies = set(studies) - set(cell_type_to_chosen_studies[term])
            if len(candidate_studies) > 0:
                cell_type_to_chosen_studies[term].append(
                    sorted(candidate_studies)[0]
                )

    term_to_chosen_exps = defaultdict(lambda: set())
    for term, studies in cell_type_to_chosen_studies.iteritems():
        for study in studies:
            candidate_exps = sorted([
                exp 
                for exp in study_to_exps[study]
                if term in set(exp_to_ms_cell_types[exp])
            ])
            if len(candidate_exps) > 10:
                chosen_exps = candidate_exps[:10]
            else:
                chosen_exps = candidate_exps
        term_to_chosen_exps[term].update(chosen_exps)

    all_chosen_studies = set()
    for chosen_studies in cell_type_to_chosen_studies.values():
        all_chosen_studies.update(chosen_studies)

    all_chosen_exps = set()
    for term, exps in term_to_chosen_exps.iteritems():
        all_chosen_exps.update(exps)
    all_chosen_exps = list(all_chosen_exps)
    include_experiments = all_chosen_exps

    print "Finally selected %d experiments" % len(include_experiments)
    print "Finally selected %d studies" % len(all_chosen_studies)
    print "Term: No. studies:"
    for term, studies in cell_type_to_chosen_studies.iteritems():
        print "\t%s: %d" % (term, len(studies))

    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "test_experiments",
                "description": "These are a subset of the experiments in the experiment list '%s' in which I attempt to grab two studies for each cell type as well as 10 experiments from each of these two studies. This will provide a small dataset on which I will be able to test the code for leave-study-out cross-validation methods without having to load the entire dataset." % (
                    parent_exp_list_name
                ),
                "experiments": list(include_experiments)
            },
            indent=4
        ))    


if __name__ == "__main__":
    main()
