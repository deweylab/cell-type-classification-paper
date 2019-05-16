#####################################################################################
#   This creates a list of test experiments to be used to debug various programs used
#   in my analyses/experiments.  This experiment list consists of an extremely small
#   set of cell types (3) and studies (6).
#####################################################################################

from optparse import OptionParser
import sys
import json
from collections import defaultdict 

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/graph_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import labelizers
import the_ontology
import kallisto_quantified_data_manager_hdf5 as kqdm

STUDY_TERM_PAIRS = [
    ("SRP002105", "CL:0000624"), # CD4-positive, alpha-beta T cell
    ("SRP076627", "CL:0000624"), # CD4-positive, alpha-beta T cell
    ("SRP081574", "CL:0000625"), # CD8-positive, alpha-beta T cell
    ("SRP094414", "CL:0000625"), # CD8-positive, alpha-beta T cell
    ("SRP053246", "CL:0000019"), # sperm
    ("SRP041833", "CL:0000019")  # sperm
]

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    exp_info_f = args[0]
    out_f = options.out_file

    og = the_ontology.the_ontology()

    with open(exp_info_f, 'r') as f:
        exp_to_info = json.load(f)

    include_experiments = set()
    exp_to_study = {
        exp: exp_to_info[exp]['study_accession']
        for exp in exp_to_info
    }
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.iteritems():
        study_to_exps[study].add(exp)
    study_to_exps = dict(study_to_exps)    
    exp_to_ms_cell_types = labelizers.cell_type_labelizer(
        exp_to_info.keys(),
        exp_to_info
    )
    include_experiments = set()
    for study_term in STUDY_TERM_PAIRS:
        study = study_term[0]
        term = study_term[1]
        for exp in study_to_exps[study]:
            if term in exp_to_ms_cell_types[exp]:
                include_experiments.add(exp)    

    print "Finally selected %d experiments" % len(include_experiments)
    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "toy",
                "description": "This is a very small toy dataset to be used for debugging purposes",
                "experiments": list(include_experiments)
            },
            indent=4
        ))    


if __name__ == "__main__":
    main()
