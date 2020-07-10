from optparse import OptionParser
import pandas as pd
import json
from os.path import join

from graph_lib.graph import DirectedAcyclicGraph
from common import the_ontology

QUALIFIER_TERMS = set([
    'CL:2000001',   # peripheral blood mononuclear cell
    'CL:0000081',   # blood cell
    'CL:0000080',   # circulating cell
    'CL:0002321'    # embryonic cell
])

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Output directory")
    (options, args) = parser.parse_args()
    
    binary_results_f = args[0]
    results_f = args[1]
    label_graph_f = args[2]
    decision_boundary_f = args[3]
    precision_thresh = float(args[4])
    out_dir = options.out_dir

    binary_results_df = pd.read_csv(binary_results_f, sep='\t', index_col=0)
    results_df = pd.read_csv(results_f, sep='\t', index_col=0) 
    decision_df = pd.read_csv(decision_boundary_f, sep='\t', index_col=0)

    # Load the ontology
    og = the_ontology.the_ontology()

    # Load the label graph
    with open(label_graph_f, 'r') as f:
        label_data = json.load(f)
    label_graph = DirectedAcyclicGraph(label_data['label_graph'])
    label_to_name = {
        x: og.id_to_term[x].name
        for x in label_graph.get_all_nodes()
    }

    label_to_f1 = {
        label: decision_df.loc[label]['F1-score']
        for label in decision_df.index
    }
    label_to_prec = {
        label: decision_df.loc[label]['precision']
        for label in decision_df.index
    }
    label_to_thresh = {
        label: decision_df.loc[label]['empirical_threshold']
        for label in decision_df.index
    }

    # Map each label to its ancestors
    label_to_ancestors = {
        label: label_graph.ancestor_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    # Filter labels according to empiracle precision
    hard_labels = set([
        label
        for label, prec in label_to_prec.items()
        if prec < precision_thresh
    ])
    
    # Map each experiment to its predicted terms
    print('Mapping each sample to its predicted labels...')
    consider_labels  = set(binary_results_df.columns) - hard_labels
    exp_to_pred_labels = {
        exp: [
            label
            for label in consider_labels
            if binary_results_df.loc[exp][label] == 1
        ]
        for exp in binary_results_df.index
    }

    print('Computing the most-specific predicted labels...')
    exp_to_ms_pred_labels = {
        exp: label_graph.most_specific_nodes(set(pred_labels) - QUALIFIER_TERMS)
        for exp, pred_labels in exp_to_pred_labels.items()
    }
 
    # Select cells with highest probability
    exp_to_select_pred_label = {
        exp: max(
            [
                (label, results_df.loc[exp][label])
                for label in ms_pred_labels
            ],
            key=lambda x: x[1]
        )[0]
        for exp, ms_pred_labels in exp_to_ms_pred_labels.items()
        if len(ms_pred_labels) > 0
    } 
   
    exp_to_update_pred = {}
    for exp, select_label in exp_to_select_pred_label.items():
        print('{}: {}'.format(exp, og.id_to_term[select_label].name))
        all_labels = label_to_ancestors[select_label] 
        exp_to_update_pred[exp] = all_labels
    
    # Add qualifier cell types
    for exp in exp_to_update_pred:
        for qual_label in QUALIFIER_TERMS:
            if qual_label in exp_to_pred_labels[exp]:
                all_labels = label_to_ancestors[qual_label]
                exp_to_update_pred[exp].update(all_labels)
 
    # Create dataframe with filtered results
    da = []
    for exp in binary_results_df.index:
        row = []
        for label in binary_results_df.columns:
            if label in exp_to_update_pred[exp]:
                row.append(1)
            else:
                row.append(0)
        da.append(row)

    df = pd.DataFrame(
        data=da,
        columns=binary_results_df.columns,
        index=binary_results_df.index
    )
    df.to_csv(join(out_dir, 'filtered_binary_classification_results.prec_{}.tsv'.format(str(precision_thresh))), sep='\t')


def select_best_most_specific(): 

    og = the_ontology.the_ontology()

if __name__ == "__main__":
    main()
