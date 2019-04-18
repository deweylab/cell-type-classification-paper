####################################################################################
#
####################################################################################

from optparse import OptionParser
import sys
import json

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/graph_lib")

import project_data_on_ontology as pdoo
import graph
from the_ontology import the_ontology

BLACKLIST = set([
    "CL:0000010",   # cultured cell
    "CL:0000578",   # experimentally modified cell in vitro
    "CL:0001034"    # cell in vitro
])

IGNORE = set([
    "CL:2000001",   # peripheral blood mononuclear cells
    "CL:0000842",   # mononuclear cell
    "CL:0000081"    # blood cell
])

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    single_cell_exp_list_f = args[0]
    bulk_exp_list_f = args[1]
    single_cell_label_graph_f = args[2]
    bulk_label_graph_f = args[3]

    out_f = options.out_file    

    og = the_ontology()

    with open(single_cell_exp_list_f, 'r') as f:
        sc_experiments_data = json.load(f)
    with open(bulk_exp_list_f, 'r') as f:
        bulk_experiments_data = json.load(f)
    
    single_cell_exps = set(sc_experiments_data['experiments'])
    single_cell_exp_list_name = sc_experiments_data['list_name']
    bulk_exps = set(bulk_experiments_data['experiments'])
    bulk_exp_list_name = bulk_experiments_data['list_name']
    assert single_cell_exp_list_name == "untampered_single_cell_primary_cells_with_data"
    assert bulk_exp_list_name == "untampered_bulk_primary_cells_with_data"


    with open(single_cell_label_graph_f, 'r') as f:
        labels_data = json.load(f)
        jsonable_graph = labels_data['label_graph']
        jsonable_exp_to_labels = labels_data['labelling']
    single_cell_label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    single_cell_exp_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_exp_to_labels)
    single_cell_exp_to_labels = {k:set(v) for k,v in single_cell_exp_to_labels.iteritems()}
    for exp, labels in single_cell_exp_to_labels.iteritems():
        rem = set()
        for label in labels:
            if len(BLACKLIST & label) > 0:
                rem.add(label) 
        labels = labels - rem

    with open(bulk_label_graph_f, 'r') as f:
        labels_data = json.load(f)
        jsonable_graph = labels_data['label_graph']
        jsonable_exp_to_labels = labels_data['labelling']
    bulk_label_graph = pdoo.graph_from_jsonable_dict(jsonable_graph)
    bulk_exp_to_labels = pdoo.labelling_from_jsonable_dict(jsonable_exp_to_labels)
    bulk_exp_to_labels = {k:set(v) for k,v in bulk_exp_to_labels.iteritems()}
    for exp, labels in bulk_exp_to_labels.iteritems():
        rem = set()
        for label in labels:
            if len(BLACKLIST & label) > 0:
                rem.add(label)
        labels = labels - rem

    all_label_sets = set([frozenset(x) for x in bulk_exp_to_labels.values()])
    #for labels in bulk_exp_to_labels.values():
    #    ms_labels = bulk_label_graph.most_specific_nodes(labels)
    #    rem = set()
    #    for label in ms_labels:
    #        if len(IGNORE & label) > 0:
    #            rem.add(label)
    #    ms_labels = ms_labels - rem
    #    all_label_sets.add(frozenset(ms_labels))

    labels_sets_not_in_bulk = set()
    removed_exps = set()
    include_experiments = set()
    for exp, labels in single_cell_exp_to_labels.iteritems():
        ms_labels = single_cell_label_graph.most_specific_nodes(labels)
        rem = set()
        for label in ms_labels:
            if len(IGNORE & label) > 0:
                rem.add(label)
        ms_labels = ms_labels - rem
        found = False
        for label_set in all_label_sets:
            if ms_labels < label_set:
                include_experiments.add(exp)
                found = True
                break
        if not found:
            labels_sets_not_in_bulk.add(frozenset(ms_labels))
            removed_exps.add(exp)

    print "%d single-cell experiments were removed" % len(removed_exps)
    print "Labels sets that were removed"
    for label_set in labels_sets_not_in_bulk:
        print [pdoo.convert_node_to_name(x, og) for x in label_set]

    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "untampered_single_cell_primary_cells_with_data_cell_types_in_bulk",
                "description": "These are all experiments that are in the experiment list '%s' and also share the same set of most-specific labels with at least one experiment in %s" % (
                    single_cell_exp_list_name,
                    bulk_exp_list_name
                ), 
                "experiments": list(include_experiments)
            },
            indent=4
        ))    
    

if __name__ == "__main__":
    main()
