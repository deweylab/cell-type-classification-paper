from optparse import OptionParser
import sys

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import load_boiler_plate_data_for_exp_list as lbpdfel

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()
    
    env_dir = args[0]
    exp_list_name = args[1]

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

    all_terms = set()
    for terms in exp_to_terms.values():
        all_terms.update(terms)

    junk_terms = set([
        "CL:0000000",   # cell
        "CL:0000003",   # native cell
        "CL:0000255",   # eukaryotic cell
        "CL:0000548",   # animal cell
        "CL:0000010",   # cultured cell
        "CL:0000578",   # experimentally modified cell in vitro
        "CL:0001034"    # cell in vitro
    ])
    terms_minus_junk = all_terms - junk_terms

    
    print "Number of experiments: %d" % len(the_exps)
    print "Number of studies: %d" % len(study_to_exps)
    print "Number of cell types: %d" % len(all_terms)
    print "Number of cell types without trivial terms: %d" % len(terms_minus_junk)

if __name__ == "__main__":
    main()
