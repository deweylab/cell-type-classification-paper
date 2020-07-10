from collections import deque, defaultdict

from onto_lib_py3 import ontology_graph
from graph_lib.graph import DirectedAcyclicGraph

def ontology_subgraph_spanning_terms(
        span_terms,
        og
    ):
    """
    Builds the ontology subgraph spanning a set of terms.
    """
    # Get most general terms
    most_general_terms = ontology_graph.most_specific_terms(
        span_terms,
        og,
        sup_relations=["inv_is_a", "inv_part_of"]
    )
    q = deque(most_general_terms)
    subgraph_source_to_targets = defaultdict(lambda: set())
    relations = ["inv_is_a", "inv_part_of"]
    #visited_ids = set(most_general_terms)
    while len(q) > 0:
        source_t_id = q.popleft()
        for rel in relations:
            if rel in og.id_to_term[source_t_id].relationships:
                for target_t_id in og.id_to_term[source_t_id].relationships[rel]:
                    target_descendants = set(
                        og.recursive_relationship(target_t_id, relations)
                    )
                    # There exists a descendant of the target represented in the samples
                    if len(target_descendants.intersection(span_terms)) > 0:
                        subgraph_source_to_targets[source_t_id].add(target_t_id)
                        q.append(target_t_id)
        #visited_ids.add(source_t_id)
    return DirectedAcyclicGraph(subgraph_source_to_targets)

