from optparse import OptionParser
import h5py
import json

from common import the_ontology
from common import ontology_utils
from graph_lib import graph
from onto_lib_py3 import ontology_graph

DATA_SET_TO_TARGET_TERM = {
    'CD19_B_cells': 'CL:0000236',                  # B cell
    'CD14_monocytes': 'CL:0001054',                 # CD14-positive monocyte
    'CD34_cells':  'CL:0008001',                    # hematopoietic precursor cell
    'CD4_CD25_regulatory_T_cells': 'CL:0000792',    # CD4-positive, CD25-positive, alpha-beta regulatory T cell 
    'CD4_CD45RA_naive_T_cells': 'CL:0000895',       # naive thymus-derived CD4-positive, alpha-beta T cell
    'CD4_CD45RO_memory_T_cells': 'CL:0000897',      # CD4-positive, alpha-beta memory T cell
    'CD4_helper_T_cells': 'CL:0000492',             # CD4-positive helper T cell
    'CD56_NK_cells': 'CL:0000623',                  # natural killer cell
    'CD8_CD45RA_naive_T_cells':  'CL:0000900',      # naive thymus-derived CD8-positive, alpha-beta T cell
    'CD8_T_cells': 'CL:0000625'                     # CD8-positive, alpha-beta T cell
}

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="File to write labels data")
    (options, args) = parser.parse_args()

    dataset_f = args[0]
    out_f = options.out_file

    # Load the ontology
    og = the_ontology.the_ontology()
    
    # Load the cell_ids and 10x datasets from which they
    # originate
    with h5py.File(dataset_f, 'r') as f:
        cell_ids = [
            str(x)[2:-1]
            for x in f['experiment'][:]
        ] 
        datasets = [
            str(x)[2:-1]
            for x in f['dataset'][:]
        ]

    # Label each cell
    cell_id_to_labels = {}
    all_labels = set()
    for dataset, cell_id in zip(datasets, cell_ids):
        ms_label = DATA_SET_TO_TARGET_TERM[dataset]
        labels = sorted(og.recursive_superterms(ms_label))
        cell_id_to_labels[cell_id] = labels
        all_labels.update(labels)

    # Generate label-graph
    label_graph = ontology_utils.ontology_subgraph_spanning_terms(
        all_labels,
        og
    )
    label_graph = graph.transitive_reduction_on_dag(label_graph)

    # Write output
    with open(out_f, 'w') as  f:
        f.write(json.dumps(
            {
                'labels_config': {},
                'label_graph': {
                    source: list(targets)
                    for source, targets in label_graph.source_to_targets.items()
                },
                'labels': cell_id_to_labels
            },
            indent=4,
            separators=(',', ': ')
        ))



if __name__ == "__main__":
    main()
