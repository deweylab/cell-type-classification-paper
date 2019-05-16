from optparse import OptionParser
import graph_lib
from graph_lib import graph
from collections import defaultdict
import math
import unittest
import numpy as np
import json
import subprocess

import bayes_net_correction_gibbs as bncg

BAYES_NET_POS = "assigned"
BAYES_NET_NEG = "unassigned"
LATENT_VAR_SPACE = [BAYES_NET_POS, BAYES_NET_NEG]

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()
    
    test()

class TestBNC(unittest.TestCase):
    def test__est_priors_from_counts(self):
        model = gibbs_sampling_model()
        label_graph = toy_small()
        item_to_labels = items_1_toy_small()
        
        label_to_p_pos = bncg._est_priors_from_counts(
            item_to_labels
        )
        self.assertEqual(label_to_p_pos['A'], 1.0)
        self.assertEqual(label_to_p_pos['B'], 6.0/12.0)
        self.assertEqual(label_to_p_pos['C'], 8.0/12.0)
        self.assertEqual(label_to_p_pos['D'], 2.0/12.0)
        self.assertEqual(label_to_p_pos['E'], 2.0/12.0)
        self.assertEqual(label_to_p_pos['F'], 3.0/12.0)
   
    def test__est_priors_from_cond_counts(self):
        model = gibbs_sampling_model()
        label_graph = toy_small()
        item_to_labels = items_1_toy_small()

        label_to_p_pos = bncg._est_priors_from_cond_counts(
            label_graph, 
            item_to_labels
        )
        self.assertEqual(label_to_p_pos['A'], 1.0)
        self.assertEqual(label_to_p_pos['B'], 2.0/8.0)   
        self.assertEqual(label_to_p_pos['C'], 3.0/7.0)
        self.assertEqual(label_to_p_pos['D'], 0.5)
        self.assertEqual(label_to_p_pos['E'], 0.5)
        self.assertEqual(label_to_p_pos['F'], 0.5)

    def test__even_space_bins(self):
        values = [0.0, 2.0, 10.0]
        n_bins = 5
        bin_bounds = bncg._even_space_bins(values, n_bins)
        true_bin_bounds = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertEqual(len(bin_bounds), len(true_bin_bounds))
        epsilon = 0.001
        for i in range(len(true_bin_bounds)):
            self.assertTrue(
                abs(bin_bounds[i]-true_bin_bounds[i]) < epsilon
            )

    def test__compute_bin_probs(self):
        bin_bounds = [-2.0, 0.0, 2.0, 4.0]
        scores = [
            -10.0,
            -1.0,
            -1.5,
            2.5,
            2.5,
            4.0,
            6.0,
            7.0
        ]
        pseudocount = 1.0
        probs = bncg._compute_bin_probs(
            bin_bounds,
            scores,
            pseudocount
        )

        true_probs = [
            2.0/13.0,
            3.0/13.0,
            1.0/13.0,
            3.0/13.0,
            4.0/13.0
        ]
        self.assertItemsEqual(probs, true_probs)

    def test__compute_bin_index(self):
        bin_bounds = [-2.0, 0.0, 2.0, 4.0]
        ind1 = bncg._compute_bin_index(bin_bounds, -6.0)
        ind2 = bncg._compute_bin_index(bin_bounds, -1.0)
        ind3 = bncg._compute_bin_index(bin_bounds, 0.0)
        ind4 = bncg._compute_bin_index(bin_bounds, 10.0)
        self.assertEqual(ind1, 0)
        self.assertEqual(ind2, 1)
        self.assertEqual(ind3, 2)
        self.assertEqual(ind4, 4)
        

    def test__histogram_estimator_pdf(self):
        bin_bounds = [-2.0, 0.0, 2.0, 4.0]
        densities = [1.0, 2.0, 3.0]
        d1 = bncg._histogram_estimator_pdf(bin_bounds, densities, -1.0)
        d2 = bncg._histogram_estimator_pdf(bin_bounds, densities, -2.0)
        d3 = bncg._histogram_estimator_pdf(bin_bounds, densities, 3.0)
        self.assertEqual(d1, math.log(1.0))
        self.assertEqual(d2, math.log(1.0))
        self.assertEqual(d3, math.log(3.0))

    def test__compute_full_conditional(self):
        r = toy_small_bn_1()
        latent_rv_to_parents = r[0]  
        latent_rv_to_children = r[1]
        latent_rv_to_svm_score_var = r[2]
        label_to_p_pos = r[3]
        svm_var_to_pos_prob = r[4]
        svm_var_to_neg_prob =r[5]

        latent_var = 'C'
        trivial_labels = set(['A'])
        neg_rvs = set([
            'E', 'F'
        ])
        c1 = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c1, 0.1724137931034483)

        latent_var = 'C'
        trivial_labels = set(['A'])
        neg_rvs = set([
            'E', 'F', 'D', 'B'
        ])
        c2 = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c2, 0.2577319587628866)

        latent_var = 'E'
        trivial_labels = set(['A'])
        neg_rvs = set([
            'F', 'D'
        ])
        c = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c, 0.8663366336633663)


        latent_var = 'E'
        trivial_labels = set(['A'])
        neg_rvs = set([
            'F'
        ])
        c = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c, 0.8536585365853658)

        latent_var = 'E'
        trivial_labels = set(['A'])
        neg_rvs = set()
        c = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c, 0.5384615384615385)

        latent_var = 'C'
        trivial_labels = set(['A'])
        neg_rvs = set()
        c = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c, 1.0)
        
        latent_var = 'A'
        trivial_labels = set(['A'])
        neg_rvs = set()
        c3 = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c3, 1.0)

        latent_var = 'E'
        trivial_labels = set('A')
        neg_rvs = set(['A', 'B', 'C', 'D', 'F'])
        c = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c, 0.0)

        r = toy_small_bn_2()
        latent_rv_to_parents = r[0]
        latent_rv_to_children = r[1]
        latent_rv_to_svm_score_var = r[2]
        label_to_p_pos = r[3]
        svm_var_to_pos_prob = r[4]
        svm_var_to_neg_prob =r[5]

        latent_var = 'A'
        trivial_labels = set(['A'])
        neg_rvs = set(['B', 'C', 'D', 'E', 'F'])
        c5 = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c5, 0.6)

        latent_var = 'A'
        trivial_labels = set(['A'])
        neg_rvs = set(['C', 'D', 'E', 'F'])
        c6 = bncg._compute_full_conditional(
            latent_var,
            trivial_labels,
            latent_rv_to_parents,
            latent_rv_to_children,
            latent_rv_to_svm_score_var,
            label_to_p_pos,
            neg_rvs,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        )
        self.assertEqual(c6, 1.0)


    def test__update_gibbs_sampling_state(self):
        r = toy_small_bn_1()
        latent_rv_to_children = r[1]
        bayes_net_dag = graph.DirectedAcyclicGraph(latent_rv_to_children)
       
        pos_rvs = set(['A', 'B'])
        skip_sampling = set(['F', 'E'])
        curr_rv = 'C'
        gibbs_sample = 1
        from_pos_to_neg = set()
        from_neg_to_pos = set()
        bncg._update_gibbs_sampling_state(
            curr_rv,
            gibbs_sample,
            pos_rvs,
            skip_sampling,
            from_pos_to_neg,
            from_neg_to_pos,
            bayes_net_dag
        )
        correct_pos_rvs = frozenset(['A', 'B', 'C'])
        correct_skip_sampling = frozenset(['A'])
        correct_from_pos_to_neg = frozenset()
        correct_from_neg_to_pos = frozenset(['C'])
        self.assertEqual(frozenset(pos_rvs), correct_pos_rvs)
        self.assertEqual(frozenset(skip_sampling), correct_skip_sampling)
        self.assertEqual(frozenset(from_neg_to_pos), correct_from_neg_to_pos)
        self.assertEqual(frozenset(from_pos_to_neg), correct_from_pos_to_neg)
        
        pos_rvs = set(['A'])
        skip_sampling = set(['D', 'F', 'E'])
        curr_rv = 'C'
        gibbs_sample = 1
        from_pos_to_neg = set()
        from_neg_to_pos = set()
        bncg._update_gibbs_sampling_state(
            curr_rv,
            gibbs_sample,
            pos_rvs,
            skip_sampling,
            from_pos_to_neg,
            from_neg_to_pos,
            bayes_net_dag
        )
        correct_pos_rvs = frozenset(['A', 'C'])
        correct_skip_sampling = frozenset(['A', 'D', 'E'])
        correct_from_pos_to_neg = frozenset()
        correct_from_neg_to_pos = frozenset(['C'])
        self.assertEqual(frozenset(pos_rvs), correct_pos_rvs)
        self.assertEqual(frozenset(skip_sampling), correct_skip_sampling)
        self.assertEqual(frozenset(from_neg_to_pos), correct_from_neg_to_pos)
        self.assertEqual(frozenset(from_pos_to_neg), correct_from_pos_to_neg)

    
    def test__same_output_as_SMILE_implementation(self):

        model = gibbs_sampling_model()
        label_graph = toy_small()
        item_to_labels = items_1_toy_small()
        # all items are trivially in their own group
        item_to_group = {
            item: item
            for item in item_to_labels
        }
        train_vecs = np.random.randn(len(item_to_labels), 10) 
        train_items = sorted(item_to_labels.keys())
        test_vecs = np.random.randn(2, 10)
        model.fit(
            train_vecs, 
            train_items, 
            item_to_labels, 
            label_graph,
            item_to_group=item_to_group
        )

        ursa_net_f = "./tmp/bayes_network.xdsl"
        ursa_bundle_f = "./tmp/bayes_network_bundle.json"
        _test_build_net_SMILE(
            model.svm_score_var_to_pos_distr,
            model.svm_score_var_to_neg_distr,
            model._svm_score_var_name,
            label_graph,
            model.label_to_p_pos,
            label_graph.get_all_nodes(),
            model.trivial_labels,
            ursa_net_f,
            ursa_bundle_f
        )

        label_to_scores = bncg._compute_scores(
            test_vecs,
            model.label_to_classifier
        )
        r = bncg._compute_evidence_cond_probs(
            label_to_scores,
            model._svm_score_var_name,
            model.svm_score_var_to_bin_bounds,
            model.svm_score_var_to_pos_distr,
            model.svm_score_var_to_neg_distr
        )
        svm_var_to_pos_probs = r[0]
        svm_var_to_neg_probs = r[1]
        svm_var_to_scores = r[2]
      
        #_exact_inference_on_smaller(
        #    svm_var_to_scores,
        #    model.svm_score_var_to_bin_bounds,
        #    model.svm_score_var_to_pos_distr,
        #    model.svm_score_var_to_neg_distr,
        #    model.label_to_p_pos
        #)


        # Model label to marginals
        SMILE_label_to_marginals = _def_test_predict_SMILE(
            svm_var_to_scores,
            model.svm_score_var_to_bin_bounds,
            ursa_net_f
        )
        print "SMILE label to marginals:"
        print SMILE_label_to_marginals

        gibbs_label_to_marginals = model._compute_marginals(
            svm_var_to_scores,
            svm_var_to_pos_probs,
            svm_var_to_neg_probs
        )
        print "Gibbs label to marginals (smart):"
        print gibbs_label_to_marginals

        #gibbs_label_to_marginals = model._compute_marginals(
        #    svm_var_to_scores,
        #    svm_var_to_pos_probs,
        #    svm_var_to_neg_probs,
        #    naive=True
        #)
        #print "Gibbs label to marginals (naive):"
        #print gibbs_label_to_marginals

        self.assertEqual(len(SMILE_label_to_marginals), len(gibbs_label_to_marginals))
        epsilon = 0.02 
        for i in range(len(SMILE_label_to_marginals)):
            self.assertEqual(
                frozenset(SMILE_label_to_marginals[i].keys()),
                frozenset(gibbs_label_to_marginals[i].keys())
            )
            for label in SMILE_label_to_marginals[i]:
                self.assertTrue(
                    abs(SMILE_label_to_marginals[i][label] - gibbs_label_to_marginals[i][label]) < epsilon
                )

        #############################################
        #   Run the tests on a larger DAG
        #############################################
        model = gibbs_sampling_model()
        label_graph = toy_medium()
        item_to_labels = items_1_toy_medium()
        # all items are trivially in their own group
        item_to_group = {
            item: item
            for item in item_to_labels
        }

        train_vecs = np.random.randn(len(item_to_labels), 10)
        train_items = sorted(item_to_labels.keys())
        test_vecs = np.random.randn(2, 10)
        model.fit(
            train_vecs,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group
        )

        ursa_net_f = "./tmp/bayes_network.xdsl"
        ursa_bundle_f = "./tmp/bayes_network_bundle.json"
        _test_build_net_SMILE(
            model.svm_score_var_to_pos_distr,
            model.svm_score_var_to_neg_distr,
            model._svm_score_var_name,
            label_graph,
            model.label_to_p_pos,
            label_graph.get_all_nodes(),
            model.trivial_labels,
            ursa_net_f,
            ursa_bundle_f
        )

        label_to_scores = bncg._compute_scores(
            test_vecs,
            model.label_to_classifier
        )
        r = bncg._compute_evidence_cond_probs(
            label_to_scores,
            model._svm_score_var_name,
            model.svm_score_var_to_bin_bounds,
            model.svm_score_var_to_pos_distr,
            model.svm_score_var_to_neg_distr
        )
        svm_var_to_pos_probs = r[0]
        svm_var_to_neg_probs = r[1]
        svm_var_to_scores = r[2]

        # Model label to marginals
        SMILE_label_to_marginals = _def_test_predict_SMILE(
            svm_var_to_scores,
            model.svm_score_var_to_bin_bounds,
            ursa_net_f
        )
        print "SMILE label to marginals:"
        print SMILE_label_to_marginals

        gibbs_label_to_marginals = model._compute_marginals(
            svm_var_to_scores,
            svm_var_to_pos_probs,
            svm_var_to_neg_probs
        )
        print "Gibbs label to marginals (smart):"
        print gibbs_label_to_marginals

        self.assertEqual(len(SMILE_label_to_marginals), len(gibbs_label_to_marginals))
        epsilon = 0.01
        for i in range(len(SMILE_label_to_marginals)):
            self.assertEqual(
                frozenset(SMILE_label_to_marginals[i].keys()),
                frozenset(gibbs_label_to_marginals[i].keys())
            )
            for label in SMILE_label_to_marginals[i]:
                self.assertTrue(
                    abs(SMILE_label_to_marginals[i][label] - gibbs_label_to_marginals[i][label]) < epsilon
                )


def gibbs_sampling_model():
    model = bncg.PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs(
        'logistic_regression',
        {
            'penalty': 'l2',
            'penalty_weight': 1.0
        },
        assert_ambig_neg=True,
        pseudocount=1,
        #prior_pos_estimation='constant',
        prior_pos_estimation='constant',
        constant_prior_pos=0.5,
        n_burn_in=3,
        artifact_dir='./tmp',
        n_bins_default=10,
        delete_artifacts=True
    )
    return model

def junction_tree_model():
    model = bnc.PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs(
        'logistic_regression',
        {
            'penalty': 'l2',
            'penalty_weight': 1.0
        },
        assert_ambig_neg=True,
        pseudocount=1,
        prior_pos_estimation='constant',
        constant_prior_pos=0.5,
        n_burn_in=1000,
        artifact_dir='./tmp',
        n_bins_default=10,
        delete_artifacts=True
    )
    return model

def toy_smaller():
    """
            A
           / \
          B   C
    """
    source_to_targets = {
        'A': ['B', 'C'],
        'B': [],
        'C': []
    }
    return graph.DirectedAcyclicGraph(source_to_targets)    

def toy_small():
    """
            A
           / \
          B   C
         / \ / \
        D   E   F
    """
    source_to_targets = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['E', 'F'],
        'D': [],
        'E': [],
        'F': []
    }
    return graph.DirectedAcyclicGraph(source_to_targets)

def toy_medium():
    """
        A   B
       / \ / \
      C   D   E
      |  /|\ / \
      | F G H   I 
      |/|\ /|\ / \
      J K L M N   O 
    """
    source_to_targets = {
        'A': ['C', 'D'],
        'B': ['D', 'E'],
        'C': ['J'],
        'D': ['F', 'G', 'H'],
        'E': ['H', 'I'],
        'F': ['J', 'K', 'L'],
        'G': [],
        'H': ['L', 'M', 'N'],
        'I': ['N', 'O'],
        'J': [],
        'K': [],
        'L': [],
        'M': [],
        'N': [],
        'O': []
    }
    return graph.DirectedAcyclicGraph(source_to_targets)

def toy_small_bn_1():
    """
    D->e_D    E->e_E   F->e_F
     \______/ \______/
       |         |
       B->e_B    C->e_C 
        \_______/
            |
            A->e_A
    """

    onto_source_to_targets = toy_small()

    bn_graph = graph.DirectedAcyclicGraph(
        onto_source_to_targets.target_to_sources
    )
    latent_rv_to_parents = bn_graph.target_to_sources
    latent_rv_to_children = bn_graph.source_to_targets
    latent_rv_to_svm_score_var = {
        'A': 'e_A',
        'B': 'e_B',
        'C': 'e_C',
        'D': 'e_D',
        'E': 'e_E',
        'F': 'e_F',
    } 

    label_to_p_pos = {
        'A': 0.6,
        'B': 0.9,
        'C': 0.2,
        'D': 0.5,
        'E': 0.5,
        'F': 0.5
    }
    svm_var_to_pos_prob = {
        'e_A': 1.0,
        'e_B': 0.8,
        'e_C': 0.5,
        'e_D': 0.1,
        'e_E': 0.7,
        'e_F': 0.2
    }
    svm_var_to_neg_prob = {
        'e_A': 0.1,
        'e_B': 0.2,
        'e_C': 0.6,
        'e_D': 0.9,
        'e_E': 0.6,
        'e_F': 0.9
    }
    return (
        latent_rv_to_parents,
        latent_rv_to_children,
        latent_rv_to_svm_score_var,
        label_to_p_pos,
        svm_var_to_pos_prob,
        svm_var_to_neg_prob
    ) 


def toy_small_bn_2():
    """
    D->e_D    E->e_E   F->e_F
     \______/ \______/
       |         |
       B->e_B    C->e_C 
        \_______/
            |
            A
    """

    onto_source_to_targets = toy_small()

    bn_graph = graph.DirectedAcyclicGraph(
        onto_source_to_targets.target_to_sources
    )
    latent_rv_to_parents = bn_graph.target_to_sources
    latent_rv_to_children = bn_graph.source_to_targets
    latent_rv_to_svm_score_var = {
        'A': 'e_A',
        'B': 'e_B',
        'C': 'e_C',
        'D': 'e_D',
        'E': 'e_E',
        'F': 'e_F',
    }

    label_to_p_pos = {
        'A': 0.6,
        'B': 0.9,
        'C': 0.2,
        'D': 0.5,
        'E': 0.5,
        'F': 0.5
    }
    svm_var_to_pos_prob = {
        'e_B': 0.8,
        'e_C': 0.5,
        'e_D': 0.1,
        'e_E': 0.7,
        'e_F': 0.2
    }
    svm_var_to_neg_prob = {
        'e_B': 0.2,
        'e_C': 0.6,
        'e_D': 0.9,
        'e_E': 0.6,
        'e_F': 0.9
    }
    return (
        latent_rv_to_parents,
        latent_rv_to_children,
        latent_rv_to_svm_score_var,
        label_to_p_pos,
        svm_var_to_pos_prob,
        svm_var_to_neg_prob
    )



def items_1_toy_small():
    label_to_items = {
        'A': [],
        'B': ['b1', 'b2'],
        'C': ['c1', 'c2', 'c3'],
        'D': ['d1', 'd2'],
        'E': ['e1', 'e2'],
        'F': ['f1', 'f2', 'f3'],
    }
    label_graph = toy_small()
    return _item_to_labels(label_to_items, label_graph)

def items_1_toy_smaller():
    label_to_items = {
        'A': [],
        'B': ['b1', 'b2'],
        'C': ['c1', 'c2', 'c3']
    }
    label_graph = toy_smaller()
    return _item_to_labels(label_to_items, label_graph)

def items_1_toy_medium():
    label_to_items = {
        'A': [],
        'B': ['b1', 'b2'],
        'C': ['c1', 'c2', 'c3'],
        'D': ['d1', 'd2', 'd3', 'd4'],
        'E': [],
        'F': ['f1'],
        'G': ['g1', 'g2'],
        'H': ['h1', 'h2', 'h3', 'h4'],
        'I': ['i1'],
        'J': ['j1', 'j2'],
        'K': ['k1', 'k2'],
        'L': ['l1', 'l2'],
        'M': ['m1', 'm2', 'm3'],
        'N': ['n1'],
        'O': ['o1', 'o2']
    }
    label_graph = toy_medium()
    return _item_to_labels(label_to_items, label_graph)

def _item_to_labels(label_to_items, label_graph):
    item_to_labels = defaultdict(lambda: set())
    for label, items in label_to_items.iteritems():
        anc_labels = label_graph.ancestor_nodes(label)
        for item in items:
            item_to_labels[item].update(anc_labels)
    print item_to_labels
    return item_to_labels


def _test_build_net_SMILE(
        svm_score_var_to_pos_distr,
        svm_score_var_to_neg_distr,
        _svm_score_var_name,
        label_graph,
        label_to_p_pos,
        all_labels,
        trivial_labels,
        ursa_net_f,
        ursa_bundle_f
    ):
    def _compute_subfactor(curr_var_index, var_to_space, all_vars):
        curr_var = all_vars[curr_var_index]
        subfactors = []
        if curr_var_index > 0:
            subsubfactors = _compute_subfactor(
                curr_var_index - 1,
                var_to_space,
                all_vars
            )
        for val in var_to_space[curr_var]:
            if curr_var_index > 0:
                for subsubfactor in subsubfactors:
                    subfactors.append(
                        subsubfactor + [val]
                    )
            else:
                subfactors.append([val])
        return subfactors


    # Each label's parents are it's children in the ontology.
    # We also create a random variable for each label that 
    # represents the SVM score for that label. This score is
    # dependent on the positive/negative assignment for that 
    # label
    latent_var_to_parents = {
        source: list(sorted(targets))
        for source, targets in label_graph.source_to_targets.iteritems()
    }
    latent_var_to_parents.update({
        label: []
        for label in all_labels
        if label not in latent_var_to_parents
    })

    label_var_to_svm_score_var = {
        var: _svm_score_var_name[var]
        for var in latent_var_to_parents
        if var not in trivial_labels
    }
    svm_score_var_to_parents = {
        label_var_to_svm_score_var[var]: [var]
        for var in latent_var_to_parents
        if var not in trivial_labels
    }

    svm_score_vars = set(svm_score_var_to_parents.keys())
    #latent_label_vars = set(latent_var_to_parents.keys())
    latent_label_vars = set(all_labels)

    # Define the possible values for each variable. We don't 
    # compute this for the SVM score variables.
    var_to_space = {
        var: LATENT_VAR_SPACE
        for var in latent_label_vars
    }

    # Compute all of the conditional probability distributions
    # for the ontology term variables
    latent_var_to_cpd = {}
    for var, parents in latent_var_to_parents.iteritems():
        factor_vars = [var] + parents
        assignments = _compute_subfactor(
            len(factor_vars)-1,
            var_to_space,
            factor_vars
        )
        cpd = []
        for assignment in assignments:
            if assignment[0] == BAYES_NET_POS:
                if BAYES_NET_POS in set(assignment[1:]):
                    cpd.append(1.0)
                else:
                    cpd.append(label_to_p_pos[var])
            elif assignment[0] == BAYES_NET_NEG:
                if BAYES_NET_POS in set(assignment[1:]):
                    cpd.append(0.0)
                else:
                    cpd.append(1.0 - label_to_p_pos[var])
        latent_var_to_cpd[var] = cpd

    assert frozenset(svm_score_var_to_pos_distr.keys()) \
        == frozenset(svm_score_var_to_neg_distr.keys())
    svm_score_var_to_cpd = {
        svm_var: svm_score_var_to_pos_distr[svm_var] + svm_score_var_to_neg_distr[svm_var]
        for svm_var in svm_score_var_to_pos_distr
    }
    svm_score_var_to_space = {
        svm_var: [
            "bin_%d" % bin_i
            for  bin_i in range(len(pos_probs))
        ]
        for svm_var, pos_probs in svm_score_var_to_pos_distr.iteritems()
    }

    ursa_json_bundle = {
        "latent_node_to_parents": latent_var_to_parents,
        "observed_node_to_parents": svm_score_var_to_parents,
        "latent_node_to_cpd": latent_var_to_cpd,
        "observed_node_to_cpd": svm_score_var_to_cpd,
        "observed_node_to_outcome_names": svm_score_var_to_space
    }
    with open(ursa_bundle_f, 'w') as f:
        f.write(json.dumps(
            ursa_json_bundle,
            indent=4,
            separators=(',', ': ')
        ))

    build_net_cmd = "run_build_bayes_net.bash %s %s" % (
        ursa_bundle_f,
        ursa_net_f
    )
    _run_cmd(build_net_cmd)


def _exact_inference_on_smaller(
        svm_var_to_scores,
        svm_score_var_to_bin_bounds,
        svm_score_var_to_pos_distr,
        svm_score_var_to_neg_distr,
        label_to_p_pos
    ):
    for q_i in range(len(list(svm_var_to_scores.values())[0])):
        evidence = {
            svm_var: scores[q_i]
            for svm_var, scores in svm_var_to_scores.iteritems()
        }

        svm_score_var_to_outcome = {}
        for svm_score_var, score in evidence.iteritems():
            bin_bounds = svm_score_var_to_bin_bounds[svm_score_var]
            bin_index = bncg._compute_bin_index(
                bin_bounds,
                score
            )
            outcome_name = "bin_%d" % bin_index
            svm_score_var_to_outcome[svm_score_var] = outcome_name


        B_bin_bounds = svm_score_var_to_bin_bounds['Evidence_B']
        C_bin_bounds = svm_score_var_to_bin_bounds['Evidence_C']
        B_bin_index = bin_index = bncg._compute_bin_index(
            B_bin_bounds,
            evidence['Evidence_B']
        )
        C_bin_index = bin_index = bncg._compute_bin_index(
            C_bin_bounds,
            evidence['Evidence_C']
        ) 
        
        svm_B_pos_prob = svm_score_var_to_pos_distr['Evidence_B'][B_bin_index]
        svm_B_neg_prob = svm_score_var_to_neg_distr['Evidence_B'][B_bin_index]
        svm_C_pos_prob = svm_score_var_to_pos_distr['Evidence_C'][C_bin_index]
        svm_C_neg_prob = svm_score_var_to_neg_distr['Evidence_C'][C_bin_index]
        
         
        numerator = label_to_p_pos['C'] * svm_C_pos_prob * (
            (svm_B_neg_prob*(1-label_to_p_pos['B'])) \
            + (svm_B_pos_prob*label_to_p_pos['B'])
        )
        denominator = numerator \
            + (1 - label_to_p_pos['C']) * svm_C_neg_prob * ( \
                ((1.0-label_to_p_pos['A'])*svm_B_neg_prob*(1-label_to_p_pos['B'])) \
                + (label_to_p_pos['A']*svm_B_pos_prob*(1-label_to_p_pos['B']))  \
                + (svm_B_pos_prob*label_to_p_pos['B'])
            )
        print "EXACT INFERENCE FOR C: %f" % (numerator / denominator)

        numerator = label_to_p_pos['B'] * svm_B_pos_prob * (
            (svm_C_neg_prob*(1-label_to_p_pos['C'])) \
            + (svm_C_pos_prob*label_to_p_pos['C'])
        )
        denominator = numerator \
            + (1 - label_to_p_pos['B']) * svm_B_neg_prob * ( \
                ((1.0-label_to_p_pos['A'])*svm_C_neg_prob*(1-label_to_p_pos['C'])) \
                + (label_to_p_pos['A']*svm_C_pos_prob*(1-label_to_p_pos['C']))  \
                + (svm_C_pos_prob*label_to_p_pos['C'])
            )
        print "EXACT INFERENCE FOR B: %f" % (numerator / denominator)

def _def_test_predict_SMILE(
        svm_var_to_scores,
        svm_score_var_to_bin_bounds,
        ursa_net_f
    ):
    label_to_marginals = []
    label_to_score_list = []
    for q_i in range(len(list(svm_var_to_scores.values())[0])):
        evidence = {
            svm_var: scores[q_i]
            for svm_var, scores in svm_var_to_scores.iteritems()
        }

        svm_score_var_to_outcome = {}
        for svm_score_var, score in evidence.iteritems():
            bin_bounds = svm_score_var_to_bin_bounds[svm_score_var]
            bin_index = bncg._compute_bin_index(
                bin_bounds,
                score
            )
            outcome_name = "bin_%d" % bin_index
            print "svm_var %s, bin %d" % (svm_score_var, bin_index)
            svm_score_var_to_outcome[svm_score_var] = outcome_name

        evidence_f = "./tmp/test_evidence.%d.json" % q_i
        with open(evidence_f, 'w')  as f:
            f.write(json.dumps(
                {
                    "observed_node_to_outcome": svm_score_var_to_outcome
                },
                indent=4,
                separators=(',', ': ')
            ))

        inference_out_f = "./tmp/test_inference_output.%d.json" % q_i
        inference_cmd = "run_bayes_net_inference.bash %s %s %s" % (
            ursa_net_f,
            evidence_f,
            inference_out_f
        )
        _run_cmd(inference_cmd)

        with open(inference_out_f, 'r') as f:
            var_to_marg = json.load(f)
        #if delete_artifacts:
        #    _run_cmd("rm %s" % evidence_f)
        #    _run_cmd("rm %s" % inference_out_f)
        label_to_marginals.append(var_to_marg)
    return label_to_marginals

def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    unittest.main()
