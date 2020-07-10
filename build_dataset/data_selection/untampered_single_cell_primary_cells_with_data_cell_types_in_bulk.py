####################################################################################
#
####################################################################################
from optparse import OptionParser
import json

from common.the_ontology import the_ontology
from graph_lib.graph import DirectedAcyclicGraph

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
    single_cell_exp_set_name = sc_experiments_data['list_name']
    bulk_exps = set(bulk_experiments_data['experiments'])
    bulk_exp_set_name = bulk_experiments_data['list_name']
    assert single_cell_exp_set_name == "untampered_single_cell_primary_cells_with_data"
    assert bulk_exp_set_name == "untampered_bulk_primary_cells_with_data"

    with open(single_cell_label_graph_f, 'r') as f:
        labels_data = json.load(f)
        sc_label_graph = labels_data['label_graph']
        sc_exp_to_labels = labels_data['labels']
    sc_exp_to_labels = {
        k: set(v) - set(BLACKLIST) 
        for k,v in sc_exp_to_labels.items()
    }

    with open(bulk_label_graph_f, 'r') as f:
        labels_data = json.load(f)
        bulk_label_graph = labels_data['label_graph']
        bulk_exp_to_labels = labels_data['labels']
    bulk_exp_to_labels = {
        k: set(v) - set(BLACKLIST)
        for k,v in bulk_exp_to_labels.items()
    }

    # The idea here is that we only want single-cell samples
    # for which ~all~ of its most-specific labels are a subset
    # of one bulk-sample's label-set. Here we collect all of the
    # unique bulk label-sets.
    #
    # For example, given a sample labeled as {embryonic cell, 
    # neural cell}, but in the bulk data we only have samples 
    # labelled as {embryonic cell} and {neural cell}. We would 
    # discard this cell.
    bulk_label_sets = set()
    for labels in bulk_exp_to_labels.values():
        bulk_label_sets.add(frozenset(labels))

    label_sets_not_in_bulk = set()
    removed_exps = set()
    include_exps = set()
    g = DirectedAcyclicGraph(sc_label_graph)
    for exp, labels in sc_exp_to_labels.items():
        ms_labels = set(g.most_specific_nodes(labels))
        ms_labels -= set(IGNORE)

        # Go through the bulk label-sets and check if the current
        # sample's set of most-specific labels is a subset of any
        # of them. If so, keep it. If not, we discard it.
        found = False
        for label_set in bulk_label_sets:
            if set(ms_labels) <= label_set:
                include_exps.add(exp)
                found = True
                break
        if not found:
            label_sets_not_in_bulk.add(frozenset(ms_labels))
            removed_exps.add(exp)

    print("{} single-cell experiments were removed".format(len(removed_exps)))
    print("Labels that were removed:")
    print(json.dumps(
        [
            [
                og.id_to_term[x].name   
                for x in label_set
            ]
            for label_set in label_sets_not_in_bulk
        ], 
        indent=True
    ))

    with open(out_f, 'w') as f:
        f.write(json.dumps(
            {
                "list_name": "untampered_single_cell_primary_cells_with_data_cell_types_in_bulk",
                "description": "These are all experiments that are in the experiment list '%s' and also share the same set of most-specific labels with at least one experiment in %s" % (
                    single_cell_exp_set_name,
                    bulk_exp_set_name
                ), 
                "experiments": list(include_exps)
            },
            indent=4
        ))    
    

if __name__ == "__main__":
    main()
