import sys

from optparse import OptionParser
import collections
from collections import defaultdict

sys.path.append("/ua/mnbernstein/projects/tbcp/data_management")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")

from map_sra_to_ontology import ontology_graph
import the_ontology as the_og


def cell_type_labelizer(
        experiment_accs, 
        exp_to_info
    ):
    og = the_og.the_ontology()
    exp_to_terms = defaultdict(lambda: set())
    for exp in experiment_accs:
        mapped_terms = set(
            exp_to_info[exp]['mapped_terms']
        )

        # compute all cell-type terms
        all_terms = set()
        for term in mapped_terms:
            all_terms.update(
                og.recursive_relationship(
                    term,
                    recurs_relationships=['is_a', 'part_of']
                )
            )
        all_terms = frozenset([
            x
            for x in all_terms
            if x.split(':')[0] == 'CL'
        ])
        exp_to_terms[exp] = all_terms
    return exp_to_terms


def most_specific_cell_type_labelizer(
        exp_accs, 
        exp_to_info
    ):
    og = the_og.the_ontology()
    exp_to_terms = defaultdict(lambda: set())
    for exp in exp_accs:
        mapped_terms = set(
            exp_to_info[exp]['mapped_terms']
        )
        # compute the most-specific cell type pertaining 
        # to each experiment
        ms_terms = ontology_graph.most_specific_terms(
            mapped_terms,
            og,
            sup_relations=['is_a', 'part_of']
        )
        ms_terms = frozenset([
            x
            for x in ms_terms
            if x.split(':')[0] == 'CL'
        ])
        exp_to_terms[exp] = ms_terms
    return exp_to_terms


def base_term_labelize(
        labelizer_name, 
        experiment_accs, 
        exp_to_info
    ):
    valid_labelizer_names = set([
        "all_cell_types",
        "most_specific_cell_types"
    ])
    assert labelizer_name in valid_labelizer_names

    if labelizer_name == "all_cell_types":
        return cell_type_labelizer(
            experiment_accs, 
            exp_to_info
        )
    elif labelizer_name == "most_specific_cell_types":
        return most_specific_cell_type_labelizer(
            experiment_accs, 
            exp_to_info
        )


if __name__ == "__main__":
    main()
