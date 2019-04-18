####################################################################################
#   TODO
####################################################################################

from optparse import OptionParser
import sys
import json
import numpy as np

sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")

import kallisto_quantified_data_manager_hdf5 as kqdm


def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    exp_to_info_f = args[0] 
    out_f = options.out_file    

    with open(exp_to_info_f, 'r') as f:
        exp_to_info = json.load(f)
    include_experiments = set(exp_to_info.keys())
    exps_w_data = set(kqdm.filter_for_experiments_in_db(include_experiments))
    include_experiments = set(include_experiments) & exps_w_data

    # Remove all experiments that have a NaN count value
    exps_list, data_matrix = kqdm.get_transcript_counts_for_experiments(
        include_experiments
    )

    found_nan_exps = set()
    for vec_i, vec in enumerate(data_matrix):
        if (vec_i + 1) % 100 == 0:
            print "Checked %d/%d vectors..." % (vec_i+1, len(data_matrix))
        sum_vec = sum(vec)
        if sum_vec == 0.0:
            print "Experiment %s has a sum of zero..." % exps_list[vec_i]
            found_nan_exps.add(exps_list[vec_i]) 
    include_experiments = include_experiments - found_nan_exps
 
    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "all_experiments_with_data",
                "description": "These are all experiments in the data set that can be found in the database.",
                "experiments": list(include_experiments)
            },
            indent=4
        ))    
   

if __name__ == "__main__":
    main()
