import sys

from optparse import OptionParser
import collections
from collections import defaultdict, deque
import json

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")

from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import graph_lib
from graph_lib import graph
from graph_lib.graph import DirectedAcyclicGraph
import ontology_graph_spanning_terms as ogst 

DEBUG = False

def convert_node_to_str(node):
    node_str = ",".join(sorted(node)) 
    return node_str
    

def convert_node_from_str(node_str):
    return frozenset(node_str.split(","))


def convert_node_to_name(node, og):
    for term in node:
        assert term in og.id_to_term
    ms_terms = ontology_graph.most_specific_terms(
        node,
        og,
        sup_relations=["is_a", "part_of"]
    )
    rep_term = sorted(ms_terms)[0]
    if len(node) > 1:
        return "%s (%d)" % (
            og.id_to_term[rep_term].name, 
            len(node)
        )
    else:
        return og.id_to_term[rep_term].name


def graph_to_jsonable_dict(collapsed_onto_graph):
    """
    Each node in the collapsed ontology graph is set of terms.
    This cannot convert to a JSON string by default. We therefore
    take each node and convert it to a string consisting of the 
    sorted terms within that node delimited by a ','.  
    """
    all_nodes = collapsed_onto_graph.get_all_nodes()
    node_to_str = {}
    for node in all_nodes:
        node_str = convert_node_to_str(node)
        node_to_str[node] = node_str
    jsonable_source_to_targets = {
        node_to_str[source]: [
            node_to_str[target]
            for target in targets
        ]
        for source, targets in collapsed_onto_graph.source_to_targets.iteritems()
    }
    return jsonable_source_to_targets


def graph_from_jsonable_dict(jsonable_source_to_targets):
    source_to_targets = {}
    for source_str, target_strs in jsonable_source_to_targets.iteritems():
        source = convert_node_from_str(source_str)
        targets = set([
            convert_node_from_str(target_str)
            for target_str in target_strs
        ])
        source_to_targets[source] = targets
    return DirectedAcyclicGraph(source_to_targets)


def labelling_to_jsonable_dict(item_to_nodes):
    item_to_strs = {
        item: [
            convert_node_to_str(node)
            for node in nodes
        ]
        for item, nodes in item_to_nodes.iteritems()
    }
    return item_to_strs


def labelling_from_jsonable_dict(item_to_node_strs):
    return {
        item: [
            convert_node_from_str(node_str)
            for node_str in node_strs
        ]
        for item, node_strs in item_to_node_strs.iteritems()
    }


def spanning_subgraph(
        og,
        terms
    ):

    # Get most general terms
    most_general_terms = ontology_graph.most_specific_terms(
        terms,
        og,
        sup_relations=["inv_is_a", "inv_part_of"]
    )
    if DEBUG:
        print "Most general terms: %s" % most_general_terms
    q = deque(most_general_terms)

    source_to_targets = defaultdict(lambda: set())
    relations = ["inv_is_a", "inv_part_of"]
    visited_ids = set(most_general_terms)
    while len(q) > 0:
        source_t_id = q.popleft()
        for rel in relations:
            if rel in og.id_to_term[source_t_id].relationships:
                for target_t_id in og.id_to_term[source_t_id].relationships[rel]:
                    target_descendants = set(
                        og.recursive_relationship(target_t_id, relations)
                    )
                    # There exists a descendant of the target represented in the items
                    if len(target_descendants & terms) > 0 and target_t_id not in visited_ids:
                        source_to_targets[source_t_id].add(target_t_id)
                        q.append(target_t_id)
    for term in terms:
        if term not in source_to_targets:
            source_to_targets[term] = set()
    return dict(source_to_targets)


def collapsed_ontology_graph_OLDEST(og, item_to_terms):
    """
    This the most naive method of collapsing the ontology that simply
    iteratively collapses nodes together that share the same data.
    """
    def _merge(
            node_a,
            node_b,
            target_to_sources,
            source_to_targets,
            node_to_items
        ):
        """
        Merge nodes represented by node_a and node_b
        """
        new_node = frozenset(node_a | node_b)
        node_a_sources = target_to_sources[node_a] - set([node_b])
        node_b_sources = target_to_sources[node_b] - set([node_a])
        node_a_targets = source_to_targets[node_a] - set([node_b])
        node_b_targets = source_to_targets[node_b] - set([node_a])
        node_a_items = node_to_items[node_a]
        node_b_items = node_to_items[node_b]

        del target_to_sources[node_a]
        del target_to_sources[node_b]
        del source_to_targets[node_a]
        del source_to_targets[node_b]
        del node_to_items[node_a]
        del node_to_items[node_b]

        for target, sources in target_to_sources.iteritems():
            if node_a in sources:
                sources.discard(node_a)
                sources.add(new_node)
            if node_b in sources:
                sources.discard(node_b)
                sources.add(new_node)
        for source, targets in source_to_targets.iteritems():
            if node_a in targets:
                targets.discard(node_a)
                targets.add(new_node)
            if node_b in targets:
                targets.discard(node_b)
                targets.add(new_node)

        target_to_sources[new_node] = set([
            source
            for source in node_a_sources
        ])
        target_to_sources[new_node].update([
            source
            for source in node_b_sources
        ])

        source_to_targets[new_node] = set([
            target
            for target in node_a_targets
        ])
        source_to_targets[new_node].update([
            target
            for target in node_b_targets
        ])

        node_to_items[new_node] = frozenset(node_a_items | node_b_items)
        return new_node

    # Gather all terms
    all_terms = set()
    for item, terms in item_to_terms.iteritems():
        all_terms.update(terms)

    # Map each term to its constituent items
    term_to_items = defaultdict(lambda: set())
    for item, terms in item_to_terms.iteritems():
        for term in terms:
            term_to_items[term].add(item)
    node_to_items = {
        frozenset([k]): frozenset(v)
        for k,v in term_to_items.iteritems()
    }

    # Map each child to its parents in the ontology
    source_to_targets = spanning_subgraph(og, all_terms)

    source_to_targets = {
        frozenset([k]): set([frozenset([x]) for x in v])
        for k,v in source_to_targets.iteritems()
    }
    target_to_sources = defaultdict(lambda: set())
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            target_to_sources[target].add(source)

    # For each leaf, repeatedly perform breadth first search
    # merging nodes on each iteration until no nodes can be
    # merged
    performed_merge = True
    while performed_merge:
        for start_node in source_to_targets.keys():
            q = deque([start_node])
            while len(q) > 0:
                target = q.popleft()
                sources = target_to_sources[target]
                performed_merge = False
                for source in sources:
                    if node_to_items[target] == node_to_items[source]:
                        new_node =_merge(
                            target,
                            source,
                            target_to_sources,
                            source_to_targets,
                            node_to_items
                        )
                        performed_merge = True
                        break
                    q.append(source)
                if performed_merge:
                    break
            if performed_merge:
                break

    # Map each item to its nodes
    item_to_nodes = defaultdict(lambda: set())
    for node, items in node_to_items.iteritems():
        for item in items:
            item_to_nodes[item].add(node)

    collapsed_onto_graph = DirectedAcyclicGraph(source_to_targets)
    return collapsed_onto_graph, item_to_nodes

def collapsed_ontology_graph_OLDER(og, most_specific_terms, item_to_terms):
    """
    This is another naive strategy for collapsing the ontology that treats the data
    for each most-specifically labelled term as unique, but then collapses nodes that
    share the same set of data as is done in the method above^^^
    """
    def _merge(
            node_a, 
            node_b, 
            target_to_sources, 
            source_to_targets, 
            node_to_items,
            node_to_colors,
        ):
        """
        Merge nodes represented by node_a and node_b
        """
        new_node = frozenset(node_a | node_b)
        node_a_sources = target_to_sources[node_a] - set([node_b])
        node_b_sources = target_to_sources[node_b] - set([node_a])
        node_a_targets = source_to_targets[node_a] - set([node_b])
        node_b_targets = source_to_targets[node_b] - set([node_a])
        node_a_items = node_to_items[node_a]
        node_b_items = node_to_items[node_b]
        node_a_colors = node_to_colors[node_a]
        node_b_colors = node_to_colors[node_b]    

        del target_to_sources[node_a]
        del target_to_sources[node_b]
        del source_to_targets[node_a]
        del source_to_targets[node_b]
        del node_to_items[node_a]
        del node_to_items[node_b]
        del node_to_colors[node_a]
        del node_to_colors[node_b]

        for target, sources in target_to_sources.iteritems():
            if node_a in sources:
                sources.discard(node_a)
                sources.add(new_node)
            if node_b in sources:
                sources.discard(node_b)
                sources.add(new_node)
        for source, targets in source_to_targets.iteritems():
            if node_a in targets:
                targets.discard(node_a)
                targets.add(new_node)
            if node_b in targets:
                targets.discard(node_b)
                targets.add(new_node)
        
        target_to_sources[new_node] = set([
            source
            for source in node_a_sources
        ])
        target_to_sources[new_node].update([
            source
            for source in node_b_sources
        ])

        source_to_targets[new_node] = set([
            target
            for target in node_a_targets
        ])
        source_to_targets[new_node].update([
            target
            for target in node_b_targets
        ])
    
        node_to_items[new_node] = frozenset(node_a_items | node_b_items)
        node_to_colors[new_node] = frozenset(node_a_colors | node_b_colors)
        return new_node

    # Gather all terms
    all_terms = set()
    for item, terms in item_to_terms.iteritems():
        all_terms.update(terms)

    # Map each term to its constituent items
    term_to_items = defaultdict(lambda: set())
    for item, terms in item_to_terms.iteritems():
        for term in terms:
            term_to_items[term].add(item)

    # Map initial set of nodes to its set of items
    node_to_items = {
        frozenset([k]): frozenset(v) 
        for k,v in term_to_items.iteritems()
    }
    
    # Map initial set of nodes to its set of colors
    node_to_colors = defaultdict(lambda: set())
    for color, term in enumerate(most_specific_terms):
        node_to_colors[frozenset([term])].add(color)
    relations = ["is_a", "part_of"]
    for term in most_specific_terms:
        for anc_term in og.recursive_relationship(term, relations):
            node_to_colors[frozenset([anc_term])].update(node_to_colors[frozenset([term])])
    
    # Map each child to its parents in the ontology
    source_to_targets = spanning_subgraph(og, all_terms)

    source_to_targets = {
        frozenset([k]): set([frozenset([x]) for x in v]) 
        for k,v in source_to_targets.iteritems()
    }
    target_to_sources = defaultdict(lambda: set())
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            target_to_sources[target].add(source)
    

    # For each leaf, repeatedly perform breadth first search
    # merging nodes on each iteration until no nodes can be
    # merged
    performed_merge = True
    while performed_merge:
        for start_node in source_to_targets.keys():
            q = deque([start_node])
            while len(q) > 0:
                target = q.popleft()
                sources = target_to_sources[target]
                performed_merge = False
                for source in sources:
                    if node_to_colors[target] == node_to_colors[source]:
                        new_node =_merge(
                            target, 
                            source, 
                            target_to_sources, 
                            source_to_targets,
                            node_to_items,
                            node_to_colors
                        )
                        performed_merge = True
                        break
                    q.append(source)
                if performed_merge:
                    break
            if performed_merge:
                break

    # Map each item to its nodes
    item_to_nodes = defaultdict(lambda: set())
    for node, items in node_to_items.iteritems():
        for item in items:
            item_to_nodes[item].add(node)

    collapsed_onto_graph = DirectedAcyclicGraph(source_to_targets)
    return collapsed_onto_graph, item_to_nodes


def collapsed_ontology_graph_OLD(og, most_specific_terms, item_to_terms):
    """
    This is a less naive, but convoluted idea to collapse the ontology
    based on 'subgraphs' that all receive the same information from some
    set of most-specific terms.
    """
    all_terms = set()
    for terms in item_to_terms.values():
        all_terms.update(terms)

    # Build a graph for the ontology 
    relations = ["inv_is_a"]
    onto_source_to_targets = defaultdict(lambda: set())
    for source_term in all_terms:
        for relation in relations:
            if not relation in og.id_to_term[source_term].relationships:
                continue
            for targ_term in og.id_to_term[source_term].relationships[relation]:
                if targ_term in all_terms:
                    onto_source_to_targets[source_term].add(targ_term)

    onto_graph = DirectedAcyclicGraph(onto_source_to_targets)
    collapsed_onto_graph = collapse_graph(most_specific_terms, onto_graph)

    term_to_items = defaultdict(lambda: set())
    for item, terms in item_to_terms.iteritems():
        for term in terms:
            term_to_items[term].add(item)
    item_to_nodes = defaultdict(lambda: set())
    for node in collapsed_onto_graph.get_all_nodes():
        for term in node:
            for item in term_to_items[term]:
                item_to_nodes[item].add(node) 
    return collapsed_onto_graph, item_to_nodes            


def collapse_graph(most_specific_terms, onto_graph):
    term_to_reachable_ms_terms = defaultdict(lambda: set())
    for source_term in onto_graph.get_all_nodes():
        reachable = set(onto_graph.descendent_nodes(source_term))
        reachable_ms_terms = reachable & set(most_specific_terms)
        term_to_reachable_ms_terms[source_term] = frozenset(reachable_ms_terms)

    # Map each set of most-specific terms that are reachable by some subset of terms
    # to that subset of terms. The values in this dictionary will be the nodes of the
    # final collapsed ontology. 
    reachable_ms_term_set_to_terms = defaultdict(lambda: set())
    for term, reachable_ms_terms in term_to_reachable_ms_terms.iteritems():
        reachable_ms_term_set_to_terms[frozenset(reachable_ms_terms)].add(term) 
    
    # Create a graph of these sets of most-specific terms where an edge is 
    # drawn from one set of most-specific terms to another if the target is
    # a subset of the first. Then, compute the transitive reduction.
    reachable_set_to_reachable_subsets = defaultdict(lambda: set())
    for source_set in reachable_ms_term_set_to_terms.keys():
        for targ_set in reachable_ms_term_set_to_terms.keys():
            if targ_set <= source_set and targ_set != source_set: 
                reachable_set_to_reachable_subsets[source_set].add(targ_set)
    for reachable_set in reachable_ms_term_set_to_terms:
        if reachable_set not in reachable_set_to_reachable_subsets:
            reachable_set_to_reachable_subsets[reachable_set] = set()
    reachable_set_to_reachable_subsets = dict(reachable_set_to_reachable_subsets)
    ms_term_set_graph = DirectedAcyclicGraph(reachable_set_to_reachable_subsets)
    ms_term_set_graph = graph_lib.graph.transitive_reduction_on_dag(ms_term_set_graph)

    # Reformulate the graph so that nodes are not sets of reachable most-specific
    # terms, but rather the set of terms for which those most-specific terms
    # are reachable 
    source_node_to_targ_nodes = {
        frozenset(reachable_ms_term_set_to_terms[source_ms_term_set]): [
                frozenset(reachable_ms_term_set_to_terms[targ_ms_term_set])
                for targ_ms_term_set in targ_ms_term_sets
            ]
        for source_ms_term_set, targ_ms_term_sets in ms_term_set_graph.source_to_targets.iteritems()
    }
    collapsed_onto_graph = DirectedAcyclicGraph(source_node_to_targ_nodes)
    return collapsed_onto_graph


def collapsed_ontology_graph(item_to_terms):
    term_to_items = defaultdict(lambda: set())
    for item, terms in item_to_terms.iteritems():
        for term in terms:
            term_to_items[term].add(item)

    # A label is a set of terms that all share the same
    # data set
    labels = set()
    item_sets_to_labels = defaultdict(lambda: set())
    for term, items in term_to_items.iteritems():
        item_sets_to_labels[frozenset(items)].add(term)
    label_to_items = {
        frozenset(labels): items
        for items, labels in item_sets_to_labels.iteritems()
    }

    # Generate graph according to the dataset subset relationships
    # between labels
    source_to_targets = defaultdict(lambda: set())
    for label_source, items_source in label_to_items.iteritems():
        for label_target, items_target in label_to_items.iteritems():
            if label_source == label_target:
                continue
            if items_target <= items_source:
                source_to_targets[label_source].add(label_target)

    label_graph = DirectedAcyclicGraph(source_to_targets)
    label_graph = graph_lib.graph.transitive_reduction_on_dag(label_graph)

    # Map each item to its labels
    item_to_labels = defaultdict(lambda: set())
    for label, items in label_to_items.iteritems():
        for item in items:
            item_to_labels[item].add(label)
    return label_graph, item_to_labels


def full_ontology_graph(og, item_to_terms):
    all_terms = set()
    for terms in item_to_terms.values():
        all_terms.update(terms)

    onto_graph = ogst.ontology_subgraph_spanning_terms(
        all_terms,
        og
    )
    label_graph_source_to_targets = {
        frozenset([source]): set([
            frozenset([target])
            for target in targets
        ])
        for source, targets in onto_graph.source_to_targets.iteritems()
    }
    label_graph = DirectedAcyclicGraph(label_graph_source_to_targets)
    label_graph = graph.transitive_reduction_on_dag(label_graph)            

    item_to_labels = defaultdict(lambda: set())
    for item, terms in item_to_terms.iteritems():
        for term in terms:
            item_to_labels[item].add(frozenset([term]))
    return label_graph, item_to_labels


def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    source_to_targets = {
        'cell': set(['native cell']),
        'native cell': set(['circulating cell', 'nucleate cell', 'somatic cell', 'eukaryotic cell', 'motile cell', 'haploid cell', 'ciliated cell']),
        'circulating cell': set(['peripheral blood mononuclear cell']),
        'somatic cell': set(['hematopoietic cell']),
        'eukaryotic cell': set(['animal cell']),
        'ciliated cell': set(['sperm']),
        'animal cell': set(['hematopoietic cell', 'germ line cell']),
        'haploid cell': set(['gamete']),
        'hematopoietic cell': set(['leukocyte']),
        'germ line cell': set(['germ cell']),
        'motile cell': set(['leukocyte', 'sperm']),    
        'nucleate cell': set(['single nucleate cell', 'lymphocyte']),
        'leukocyte': set(['nongranular leukocyte']),    
        'germ cell': set(['male germ cell', 'gamete']),
        'single nucleate cell': set(['mononuclear cell']),
        'nongranular leukocyte': set(['mononuclear cell', 'lymphocyte']),
        'male germ cell': set(['male gamete']),
        'gamete': set(['male gamete']),
        'mononuclear cell': set(['peripheral blood mononuclear cell']),
        'lymphocyte': set(['T cell']),    
        'male gamete': set(['sperm']),
        'peripheral blood mononuclear cell': set([]),
        'T cell': set(['alpha-beta T cell', 'mature T cell']),
        'sperm': set([]),
        'alpha-beta T cell': set(['mature alpha-beta T cell']),
        'mature T cell': set(['mature alpha-beta T cell']),
        'mature alpha-beta T cell': set(['CD8-positive mature alpha-beta T cell', 'CD4-positive mature alpha-beta T cell']),
        'CD8-positive mature alpha-beta T cell': set([]),
        'CD4-positive mature alpha-beta T cell': set([]),
    }
    onto_graph = DirectedAcyclicGraph(source_to_targets)
    #ms_terms = set(['peripheral blood mononuclear cell', 'CD8-positive mature alpha-beta T cell', 'CD4-positive mature alpha-beta T cell', 'sperm'])
    #collapsed_graph = collapse_graph(ms_terms, onto_graph)
    #print collapsed_graph.source_to_targets


    item_to_terms = {
        1: set(['peripheral blood mononuclear cell', 'CD4-positive mature alpha-beta T cell']),
        2: set(['CD4-positive mature alpha-beta T cell']),
        3: set(['CD8-positive mature alpha-beta T cell']),
        4: set(['sperm']),
        5: set(['peripheral blood mononuclear cell'])
    }
    item_to_terms_new = defaultdict(lambda: set())
    for item, terms in item_to_terms.iteritems():
        for term in terms:  
            item_to_terms_new[item].update(onto_graph.ancestor_nodes(term))
    item_to_terms = dict(item_to_terms_new)

    #print "ITEM TO TERMS:"
    #print item_to_terms

    label_graph = collapsed_ontology_graph_by_data(item_to_terms)
    print label_graph.source_to_targets

if __name__ == "__main__":
    main()


