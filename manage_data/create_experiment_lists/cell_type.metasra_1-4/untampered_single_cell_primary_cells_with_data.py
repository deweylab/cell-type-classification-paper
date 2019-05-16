####################################################################################
#   This creates a list of experiments corresponding to untampered bulk experiments.
#   By 'untampered' I mean all experiments that have been treated in such a way that
#   a purposeful change in gene expression was induced.  This does not include
#   experiments that have been transfected with a control vector.
####################################################################################

from optparse import OptionParser
import sys
import json

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/graph_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")

import graph
import kallisto_quantified_data_manager_hdf5 as kqdm

EXCLUDE_TAGS = set([
    "experimental_treatment",
    "in_vitro_differentiated_cells",
    "stimulation",
    "infected",
    "diseased_cells",
    "alternate_assay",
    "cell_line",
    "tissue"
])

EXCEPT_EXCLUDE_TAGS = set([
    "transduced_control",
    "transfected_control",
    "sirna_treatment_control",
    "shrna_treatment_control"
])

SINGLE_CELL_TAG = "single_cell"

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    annotation_f = args[0]
    out_f = options.out_file    

    with open(annotation_f, 'r') as f:
        annotation = json.load(f)

    tags_f = "/ua/mnbernstein/projects/tbcp/phenotyping/manage_data/tags/tags.json"
    tags_graph = import_tags_graph(tags_f)
    exclude_tags = set(EXCLUDE_TAGS)
    for ex_tag in EXCLUDE_TAGS:
        exclude_tags.update(tags_graph.ancestor_nodes(ex_tag))

    include_experiments = set()
    for annot_data in annotation['annotated_studies']:    
        for partition in annot_data['partitions']:
            include_partition = True
            for tag in partition['tags']:
                if tag not in EXCEPT_EXCLUDE_TAGS and tag in exclude_tags:
                    include_partition = False
                    break
            if include_partition and SINGLE_CELL_TAG in set(partition['tags']):
                include_experiments.update(partition['experiments'])

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
                "list_name": "untampered_single_cell_primary_cells_with_data",
                "description": "These are all experiments that are labelled with %s, but not %s, making exceptions for %s.  All of these samples can be found in the database." % (
                    SINGLE_CELL_TAG,
                    list(EXCLUDE_TAGS),
                    list(EXCEPT_EXCLUDE_TAGS)
                ), 
                "experiments": list(include_experiments)
            },
            indent=4
        ))    

def import_tags_graph(tags_f):
    with open(tags_f, 'r') as f:
        tags_data = json.load(f)
    tags_nodes = tags_data['definitions']
    tags_source_to_targets = {
        x: []
        for x in tags_nodes
    }
    tags_source_to_targets.update(tags_data['implications'])
    tags_graph = graph.DirectedAcyclicGraph(tags_source_to_targets)
    return tags_graph
    

if __name__ == "__main__":
    main()
