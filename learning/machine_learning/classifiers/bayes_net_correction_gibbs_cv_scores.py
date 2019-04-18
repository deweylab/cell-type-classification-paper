#################################################################
#   Per label SVM classifiers with Bayesian network correction
#   as described in "Ontology aware classification of tissue
#   and cell-type signals in gene expression profiles across
#   platforms and technologies" by Lee,S.K. et al. 2013
#################################################################

import sys
import os
from os.path import join

import subprocess
from optparse import OptionParser
from collections import defaultdict, deque
import numpy as np
from scipy.stats import norm
import itertools
import math
import json
import random
import cPickle

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping")

import project_data_on_ontology as pdoo

import log_p
import onto_lib
from onto_lib import ontology_graph
#from per_label_svm import PerLabelSVM
from per_label_classifier import PerLabelClassifier
import graph_lib
from graph_lib import graph
from graph_lib.graph import DirectedAcyclicGraph
import binary_classifier as bc

BAYES_NET_POS = "assigned"
BAYES_NET_NEG = "unassigned"
LATENT_VAR_SPACE = [BAYES_NET_POS, BAYES_NET_NEG]
EPSILON = 0.0000001

N_GIBBS_ITERS = 30000
DEBUG = 0

#N_GIBBS_ITERS = 2
#DEBUG = 4

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    feat_vecs = [
        [1,1,1,2,3],
        [10,23,1,24,32],
        [543,21,23,2,5]
    ]

    items = [
        'a',
        'b',
        'c'
    ]
    item_to_labels = {
        'a':['hepatocyte', 'disease'],
        'b':['T-cell'],
        'c':['stem cell', 'cultured cell']
    }
    

def _even_space_bins(values, n_bins):
    if len(values) == 1:
        return values
    max_val = max(values) + EPSILON
    min_val = min(values) - EPSILON
    width = (max_val - min_val) / float(n_bins)
    bin_bounds = []
    for i in range(n_bins):
        left = min_val + (i * width)
        bin_bounds.append(left)
    bin_bounds.append(bin_bounds[-1] + width)
    return bin_bounds


def _compute_bin_probs(
        bin_bounds,
        scores,
        pseudocount
    ):
    """
    Compute the probability of a score falling
    into each bin.
    Args:
        bin_bounds: a list of ordered numbers corresponding
            to the boundaries of each bin. The minimum value
            corresponds to the boundary for the open 
            lower bin. The maximum value corresponds to the
            boundary for the right open bin
        scores: the values that will be placed into each bin
        pseudocount: the psuedocount for the bin probability
            estimates
    """
    # counts for the open interval on the left and right
    counts_left_open = pseudocount
    counts_right_open = pseudocount

    counts_inner = [
        pseudocount
        for i in range(len(bin_bounds)-1)
    ]

    for score in scores:
        if score < bin_bounds[0]:
            counts_left_open += 1
        elif score >= bin_bounds[-1]:
            counts_right_open += 1
        else:
            for bin_i in range(len(bin_bounds)-1):
                left = bin_bounds[bin_i]
                right = bin_bounds[bin_i+1]
                if score >= left and score < right:
                    counts_inner[bin_i] += 1
    total = counts_left_open + sum(counts_inner) + counts_right_open

    prob_left_open = counts_left_open / float(total)
    prob_right_open = counts_right_open / float(total)
    probs = [
        count / float(total)
        for count in counts_inner
    ]
    return [prob_left_open] + probs + [prob_right_open]

   
def _compute_bin_index(bin_bounds, score):
    bin_index = None
    # the score falls in the left open bin
    if score < bin_bounds[0]:
        bin_index = 0
    # the score falls in the right open bin
    elif score > bin_bounds[-1]:
        bin_index = len(bin_bounds)
    else:
        for b_i in range(len(bin_bounds)-1):
            left = bin_bounds[b_i]
            right = bin_bounds[b_i+1]
            if score >= left and score < right:
                bin_index = b_i + 1
    return bin_index


def _est_priors_from_counts(item_to_labels):
        """
        Estimate the prior probability of a latent variable being
        assigned based on the counts in the data. That is, this sets
        the probability of a latent variable being assigned if none
        of its children are assigned.
        
        Args:
            item_to_labels: maps each item to its label set
        """
        label_to_p_pos = {}
        label_to_items = defaultdict(lambda: set())
        for item, labels in item_to_labels.iteritems():
            for label in labels:
                label_to_items[label].add(item)
        for label, label_items in label_to_items.iteritems():
            label_to_p_pos[label] = float(len(label_items))/len(item_to_labels)
        return label_to_p_pos


def _est_priors_from_cond_counts(
        label_graph,
        item_to_labels
    ):
    label_to_p_pos = {}
    label_to_items = defaultdict(lambda: set())
    all_items = set()
    for item, labels in item_to_labels.iteritems():
        for label in labels:
            label_to_items[label].add(item)
            all_items.add(item)
    for label, label_items in label_to_items.iteritems():
        if len(label_graph.source_to_targets[label]) == 0:
            #label_to_p_pos[label] = 0.5
            label_to_p_pos[label] = float(len(label_items))/len(item_to_labels)
        else:
            child_items = set()
            for child_label in label_graph.source_to_targets[label]:
                child_items.update(
                    label_to_items[child_label]
                )
            all_items_not_children = all_items - child_items
            label_items_not_children = label_items - child_items
            if len(all_items_not_children) == 0:
                p_pos = 1.0
            else:
                p_pos = float(len(label_items_not_children)+1.0) / (len(all_items_not_children)+1.0)
            label_to_p_pos[label] = p_pos
    return label_to_p_pos


def _compute_full_conditional(
        latent_var,
        trivial_labels, 
        latent_rv_to_parents, 
        latent_rv_to_children,
        latent_rv_to_svm_score_var, 
        label_to_p_pos, 
        neg_rvs,
        svm_var_to_pos_prob,
        svm_var_to_neg_prob
    ):
    """
    Compute the full conditional probability of a node.
    Note in all comments and variable namings, the 'parents'
    of a node are parents in the Bayesian network, but children
    in the ontology.
    """
    svm_var = latent_rv_to_svm_score_var[latent_var]

    # The probability of a classifer random variable given
    # the two possible values of its latent random variable.
    # If the latent r.v. is a trivial label, then we don't
    # want an evidence r.v. A shorthand for this
    # is to make this evidence r.v. trivially 1.0.
    if latent_var in trivial_labels:
        svm_p_pos = 1.0
        svm_p_neg = 1.0
    else:
        svm_p_pos = svm_var_to_pos_prob[svm_var]
        svm_p_neg = svm_var_to_neg_prob[svm_var]

    # The probability of the two possible assignments of the
    # latent random variable given that all parents are zero
    parents = latent_rv_to_parents[latent_var]
    if parents <= neg_rvs:
        latent_var_p_pos = label_to_p_pos[latent_var]
        latent_var_p_neg = 1.0 - label_to_p_pos[latent_var]
    else:
        latent_var_p_pos = 1.0
        latent_var_p_neg = 0.0

    # Compute the probability of all children conditional
    # on their parents. Here we compute the term for when 
    # latent_var is assigned as negative -- that is, the 
    # Python variable children_p_neg.
    children_p_neg = 1.0
    debug_str = ""
    for child in latent_rv_to_children[latent_var]:
        parents = set(latent_rv_to_parents[child])
        parents -= set([latent_var]) # We assume it's negative

        # The case in which all parents are negative
        if parents <= neg_rvs:
            # If the child is negative, and all its parents
            # are negative then the probability it too would be 
            # negative is 1.0 - the prior that it's positive
            if child in neg_rvs:
                children_p_neg *= (1.0 - label_to_p_pos[child])
                debug_str += "%s_n=%f * " % (child,  (1.0 - label_to_p_pos[child]))
            else:
                children_p_neg *= label_to_p_pos[child]
                debug_str += "%s_p=%f * " % (child,  label_to_p_pos[child])
        # The case in which there exists a positive parent
        else:
            if child in neg_rvs:
                children_p_neg *= 0.0
                debug_str += "%s_n=%f * " % (child, 0.0)
            else:
                children_p_neg *= 1.0
                debug_str += "%s_p=%f * " % (child, 1.0)

    # Compute the probability of all children conditional
    # on their parents. Here we compute the variable children_p_pos 
    # -- the term for when the latent variable is positive, but this 
    # always equals 1.0 so there is no need to compute it.
    children_p_pos = 1.0
    if len(latent_rv_to_children[latent_var] & neg_rvs) > 0:
        children_p_pos = 0.0

    debug_str = debug_str[0:-3]

    pos_term = svm_p_pos * latent_var_p_pos * children_p_pos
    neg_term = svm_p_neg * latent_var_p_neg * children_p_neg
    if pos_term == 0.0:
        result_prob = 0.0
    else:
        result_prob = pos_term / (pos_term + neg_term)
    if DEBUG > 3:
        print "Computing full conditional for node %s. Neg nodes: %s" % (latent_var, neg_rvs)
        print "Positive term: svm_p_pos=%f * latent_var_p_pos=%f * children_p_pos=%f" % (svm_p_pos, latent_var_p_pos, children_p_pos)
        print "Negative term: svm_p_neg=%f * latent_var_p_neg=%f * children_p_neg=(%s)=%f" % (svm_p_neg, latent_var_p_neg, debug_str, children_p_neg)
        print "Full conditional prob = %f" % result_prob
    return result_prob

def _update_gibbs_sampling_state(
        curr_rv,
        full_cond_sample,
        pos_rvs,
        skip_sampling,
        from_pos_to_neg, 
        from_neg_to_pos,
        bayes_net_dag 
    ):
    # Check if the node went from being set negative to being set 
    # positive. If so, then we need to perform 3 steps: 
    #   1.  This is a node for which we will need to 
    #       track and update the counter when it turns back to 
    #       negative.
    #   2.  Each child can now be skipped (all children are 
    #       positive)
    #   3.  Each parent ~must~ be negative. For each of these
    #       negative parents, check if all children are 
    #       positive. If so, these nodes must be sampled on 
    #       the next visit.
    if full_cond_sample == 1 and curr_rv not in pos_rvs:
        # Step 1
        from_neg_to_pos.add(curr_rv)
        pos_rvs.add(curr_rv)
        # Step 2
        children = bayes_net_dag.source_to_targets[curr_rv]
        skip_sampling.update(children)
        # Step 3
        parents = bayes_net_dag.target_to_sources[curr_rv]
        for parent in parents:
            children_of_parents = bayes_net_dag.source_to_targets[parent]
            if children_of_parents <= pos_rvs:
                skip_sampling.remove(parent)
        if DEBUG > 2:
            print "Node %s went from - to +" % curr_rv
            print "The positively labelled nodes are now:\n%s" % pos_rvs


    # Check if the node went from being set positive to being set 
    # negative. Note, We then must perform the following steps:
    #   1.  This is a node for which we will need to increment its 
    #       counter. 
    #   2.  All children of this node ~must~ be positive. For each
    #       of these positive children, check whether its parents
    #       are all negative now. If so, we must explicitly sample
    #       from this node on the next visit.
    #   3.  All parents of this node ~must~ be negative, otherwise,
    #       we wouldn't have sampled from this node.  These parents
    #       no longer need to be sampled because they now have a 
    #       negative child.
    elif full_cond_sample == 0 and curr_rv in pos_rvs:
        # Step 1
        from_pos_to_neg.add(curr_rv)
        pos_rvs.remove(curr_rv)
        # Step 2
        for child in bayes_net_dag.source_to_targets[curr_rv]:
            all_parents = bayes_net_dag.target_to_sources[child]
            if len(all_parents & pos_rvs) == 0:
                skip_sampling.remove(child)
        # Step 3
        parents = bayes_net_dag.target_to_sources[curr_rv]
        skip_sampling.update(parents)
        if DEBUG > 2:
            print "Node %s went from + to -" % curr_rv
            print "The positively labelled nodes are now:\n%s" % pos_rvs


def _compute_scores(queries, label_to_classifier):
        """
        Make prediction by computing the marginal distributions over
        the latent random variables in the Bayesian network. Use Gibbs
        sampling to estimate these marginals.
        """
        #print "Prior probs for each node:\n%s" % self.label_to_p_pos
        label_to_scores = {}
        for label, classifier in label_to_classifier.iteritems():
            scores = classifier.decision_function(queries)
            label_to_scores[label] = scores
        return label_to_scores

def _compute_evidence_cond_probs(
        label_to_scores, 
        _svm_score_var_name, 
        svm_score_var_to_bin_bounds, 
        svm_score_var_to_pos_distr, 
        svm_score_var_to_neg_distr
    ):
    label_to_score_list = []
    svm_var_to_pos_probs = defaultdict(lambda: [])
    svm_var_to_neg_probs = defaultdict(lambda: [])
    svm_var_to_scores = {
        _svm_score_var_name[label]: scores
        for label, scores in label_to_scores.iteritems()
    }
    for q_i in range(len(list(label_to_scores.values())[0])):
        label_to_score_list.append({
            label: scores[q_i]
            for label, scores in label_to_scores.iteritems()
        })
        for svm_var, scores in svm_var_to_scores.iteritems():
            score = scores[q_i]
            bin_bounds = svm_score_var_to_bin_bounds[svm_var]
            bin_index = _compute_bin_index(
                bin_bounds,
                score
            )

            print "SVM var: %s. Bin bound %d" % (svm_var, bin_index)

            pos_prob = svm_score_var_to_pos_distr[svm_var][bin_index]
            neg_prob = svm_score_var_to_neg_distr[svm_var][bin_index]
            svm_var_to_pos_probs[svm_var].append(pos_prob)
            svm_var_to_neg_probs[svm_var].append(neg_prob)
    svm_var_to_pos_probs = dict(svm_var_to_pos_probs)
    svm_var_to_neg_probs = dict(svm_var_to_neg_probs)
    return (
        svm_var_to_pos_probs,
        svm_var_to_neg_probs,
        svm_var_to_scores,
        label_to_score_list
    )


class PerLabelSVM_BNC_DiscBins_NaiveBayes(PerLabelClassifier):

    def __init__(
        self,
        binary_classifier_type,
        binary_classifier_params,
        assert_ambig_neg,
        pseudocount,
        n_bins_default,
        artifact_dir,
        delete_artifacts=True,
        downweight_by_group=False,
        prior_pos_estimation='constant'
    ):
        """
        Args:
        """
        super(PerLabelSVM_BNC_DiscBins_NaiveBayes, self).__init__(
            binary_classifier_type,
            binary_classifier_params,
            downweight_by_group=downweight_by_group,
            assert_ambig_neg=assert_ambig_neg
        )
        self.trivial_labels = None
        self.bayes_net = None
        self.artifact_dir = artifact_dir
        self.delete_artifacts = delete_artifacts
        self.pseudocount = pseudocount
        self.n_bins_default = n_bins_default
        self.prior_pos_estimation = prior_pos_estimation
        _run_cmd("mkdir %s" % self.artifact_dir)

    def _fit_score_distrs(
            self,
            train_vecs,
            item_to_index,
            label_var_to_svm_score_var,
            item_to_group=None
        ):
        svm_score_var_to_pos_distr = {}
        svm_score_var_to_neg_distr = {}
        svm_score_var_to_space = {}
        svm_score_var_to_bin_bounds = {}
        for label in self.label_to_pos_scores:
            pos_items = self.label_to_pos_items[label]
            neg_items = self.label_to_neg_items[label]
            
            r = _compute_folds(
                pos_items,
                neg_items,
                item_to_group
            )
            fold_1_pos = r[0]
            fold_1_neg = r[1]
            fold_2_pos = r[2]
            fold_2_neg = r[3]
            item_to_group_fold_1 = r[4]
            item_to_group_fold_2 = r[5]

            r = _compute_cross_fold_val_scores(
                fold_1_pos,
                fold_1_neg,
                fold_2_pos,
                fold_2_neg,
                item_to_group_fold_1,
                item_to_group_fold_2,
                train_vecs,
                item_to_index,
                self.binary_classifier_type,
                self.binary_classifier_params,
                self.downweight_by_group
            )
            pos_scores_fold_1 = r[0]
            neg_scores_fold_1 = r[1]
            pos_scores_fold_2 = r[2]
            neg_scores_fold_2 = r[3]

            
            pos_scores = list(pos_scores_fold_1) + list(pos_scores_fold_2)
            neg_scores = list(neg_scores_fold_1) + list(neg_scores_fold_2)
             
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                # If we were unable to estimate scores using
                # 2-fold cross-valdiation, then fall back on
                # using the training scores
                print "Falling back on training scores..."
                pos_scores = self.label_to_pos_scores[label]
                neg_scores = self.label_to_neg_scores[label]
                bin_bounds = _bin_estimation(
                    pos_items,
                    neg_items,
                    pos_scores,
                    neg_scores,
                    item_to_group,
                    self.n_bins_default
                )
            else:
                bin_bounds = _bin_estimation(
                    fold_1_pos + fold_2_pos,
                    fold_1_neg + fold_2_neg,
                    pos_scores,
                    neg_scores,
                    item_to_group,
                    self.n_bins_default
                )
            pos_probs = _compute_bin_probs(
                bin_bounds,
                pos_scores,
                self.pseudocount
            )
            neg_probs = _compute_bin_probs(
                bin_bounds,
                neg_scores,
                self.pseudocount
            )
            outcome_names = [
                "bin_%d" % bin_i
                for  bin_i in range(len(pos_probs))
            ]
            svm_score_var = self.latent_rv_to_svm_score_var[label]
            svm_score_var_to_pos_distr[svm_score_var] = pos_probs
            svm_score_var_to_neg_distr[svm_score_var] = neg_probs
            svm_score_var_to_bin_bounds[svm_score_var] = bin_bounds
        return (
            svm_score_var_to_pos_distr,     
            svm_score_var_to_neg_distr,
            svm_score_var_to_bin_bounds
        )


    def fit(
        self, 
        training_feat_vecs, 
        training_items, 
        item_to_labels, 
        label_graph,
        item_to_group=None,
        verbose=False,
        feat_names=None
    ):
        super(PerLabelSVM_BNC_DiscBins_NaiveBayes, self).fit(
            training_feat_vecs,
            training_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            feat_names=feat_names
        )

        # Estimate the prior probabilities
        if self.prior_pos_estimation == "constant":
            self.label_to_p_pos = defaultdict(lambda: constant_prior_pos)
        elif self.prior_pos_estimation == "from_counts":
            self.label_to_p_pos = _est_priors_from_counts(
                self.item_to_labels
            )
        elif self.prior_pos_estimation == "from_counts_conditional":
            self.label_to_p_pos = _est_priors_from_cond_counts(
                self.label_graph,
                self.item_to_labels
            )

        # Compute the directed graph formed by the labels
        # spanning the samples
        all_labels = set()
        for item in training_items:
            all_labels.update(self.item_to_labels[item])
        source_to_targets = self.label_graph.source_to_targets

        # Each label's parents are it's children in the label graph.
        # We also create a random variable for each label that 
        # represents the classifer score for that label. This score 
        # is dependent on the positive/negative assignment for that 
        # label
        self.latent_rv_to_parents = {
            source: set(targets)
            for source, targets in source_to_targets.iteritems()
        }
        self.latent_rv_to_parents.update({
            label: set()
            for label in all_labels
            if label not in self.latent_rv_to_parents
        })

        self.latent_rv_to_children = defaultdict(lambda: set())
        for child, parents in self.latent_rv_to_parents.iteritems():
            for parent in parents:
                self.latent_rv_to_children[parent].add(child)
        self.latent_rv_to_children.update({
            label: set()
            for label in all_labels
            if label not in self.latent_rv_to_children
        })
        self.latent_rv_to_children = dict(self.latent_rv_to_children) 

        # Generate identifiers for each SVM score random variable
        self._svm_score_var_name = {
            label: "Evidence_" + pdoo.convert_node_to_str(label)
            for label in self.latent_rv_to_parents
        }

        # Map each latent random variable to its corresponding
        # SVM score random variable
        self.latent_rv_to_svm_score_var = {
            var: self._svm_score_var_name[var]
            for var in self.latent_rv_to_parents
        }

        svm_score_var_to_parents = {
            self.latent_rv_to_svm_score_var[var]: [var]
            for var in self.latent_rv_to_parents
        }
        svm_score_var_to_parents = {
            svm_score_var: list(sorted(parents))
            for svm_score_var, parents in svm_score_var_to_parents.iteritems()
        }

        svm_score_vars = set(svm_score_var_to_parents.keys())
        latent_label_vars = set(self.latent_rv_to_parents.keys())

        # Fit the conditional distributions of the SVM score random 
        # variables conditioned on each latent random variable
        item_to_index = {
            item: i
            for i,item in enumerate(training_items)
        }
        r = self._fit_score_distrs(
            training_feat_vecs,
            item_to_index,
            self.latent_rv_to_svm_score_var,
            item_to_group
        )
        self.svm_score_var_to_pos_distr = r[0]
        self.svm_score_var_to_neg_distr = r[1]
        self.svm_score_var_to_bin_bounds = r[2] 
    
    def _compute_marginals(
        self,
        svm_var_to_scores,
        svm_var_to_pos_probs,
        svm_var_to_neg_probs
    ):
        label_to_marginals = []
        label_to_marginal_sequence_list = []
        for q_i in range(len(list(svm_var_to_scores.values())[0])):
            svm_var_to_score = {
                svm_var: scores[q_i]
                for svm_var, scores in svm_var_to_scores.iteritems()
            }
            svm_var_to_pos_prob = {
                svm_var: probs[q_i]
                for svm_var, probs in svm_var_to_pos_probs.iteritems()
            }
            svm_var_to_neg_prob = {
                svm_var: probs[q_i]
                for svm_var, probs in svm_var_to_neg_probs.iteritems()
            }
            
            latent_var_to_count = defaultdict(lambda: 0)
            bayes_net_dag = DirectedAcyclicGraph(self.latent_rv_to_children)
            all_latent_rvs = bayes_net_dag.get_all_nodes()
            rv_to_marginal = {}
            for rv in all_latent_rvs:
                svm_var = self.latent_rv_to_svm_score_var[rv]
                p_pos = self.label_to_p_pos[rv]
                if not svm_var in svm_var_to_score:
                    rv_to_marginal[rv] = 1.0
                else:
                    marginal = (svm_var_to_pos_prob[svm_var]*p_pos) / ((svm_var_to_pos_prob[svm_var]*p_pos) + (svm_var_to_neg_prob[svm_var]*(1-p_pos)))
                    rv_to_marginal[rv] = marginal
            label_to_marginals.append(rv_to_marginal)
        return label_to_marginals


    def predict(self, queries, verbose=False):
        label_to_scores = _compute_scores(
            queries,
            self.label_to_classifier
        )
        r = _compute_evidence_cond_probs(
            label_to_scores,
            self._svm_score_var_name,
            self.svm_score_var_to_bin_bounds,
            self.svm_score_var_to_pos_distr,
            self.svm_score_var_to_neg_distr
        )
        svm_var_to_pos_probs = r[0]
        svm_var_to_neg_probs = r[1]
        svm_var_to_scores = r[2]
        label_to_score_list = r[3]
        label_to_marginals = self._compute_marginals(
            svm_var_to_scores,
            svm_var_to_pos_probs,
            svm_var_to_neg_probs
        )
        return label_to_marginals, label_to_score_list


class PerLabelSVM_BNC_DiscBins_Gibbs(PerLabelClassifier):

    def __init__(
        self,
        binary_classifier_type,
        binary_classifier_params,
        assert_ambig_neg,
        pseudocount,
        prior_pos_estimation,
        n_burn_in,
        artifact_dir,
        delete_artifacts=True,
        downweight_by_group=False,
        constant_prior_pos=None
    ):
        """
        Args:
            prior_pos_estimation - method for which to estimate the 
                prior probability that a label is positive given
                all children are negative
        """
        super(PerLabelSVM_BNC_DiscBins_Gibbs, self).__init__(
            binary_classifier_type, 
            binary_classifier_params, 
            downweight_by_group=downweight_by_group,
            assert_ambig_neg=assert_ambig_neg
        )
        self.label_to_pos_var = {}
        self.label_to_neg_var = {}
        self.label_to_pos_mean = {}
        self.label_to_neg_mean = {}
        self.trivial_labels = None
        self.prior_pos_estimation = prior_pos_estimation
        self.bayes_net = None
        self.label_to_p_pos = None
        self.artifact_dir = artifact_dir
        self.delete_artifacts = delete_artifacts
        self.pseudocount = pseudocount
        self.n_burn_in = n_burn_in
        self.constant_prior_pos = constant_prior_pos
        _run_cmd("mkdir %s" % self.artifact_dir)

    def fit(
        self, 
        training_feat_vecs, 
        training_items, 
        item_to_labels, 
        label_graph,
        item_to_group=None,
        verbose=False,
        feat_names=None
    ):
        super(PerLabelSVM_BNC_DiscBins_Gibbs, self).fit(
            training_feat_vecs,
            training_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            feat_names=feat_names
        )

        # Estimate the prior probabilities
        if self.prior_pos_estimation == "constant":
            self.label_to_p_pos = defaultdict(lambda: constant_prior_pos)
        elif self.prior_pos_estimation == "from_counts":
            self.label_to_p_pos = _est_priors_from_counts(
                self.item_to_labels
            )
        elif self.prior_pos_estimation == "from_counts_conditional":
            self.label_to_p_pos = _est_priors_from_cond_counts(
                self.label_graph, 
                self.item_to_labels
            )

        # Compute the directed graph formed by the labels
        # spanning the samples
        all_labels = set()
        for item in training_items:
            all_labels.update(self.item_to_labels[item])
        source_to_targets = self.label_graph.source_to_targets

        # Each label's parents are it's children in the label graph.
        # We also create a random variable for each label that 
        # represents the classifer score for that label. This score 
        # is dependent on the positive/negative assignment for that 
        # label
        self.latent_rv_to_parents = {
            source: set(targets)
            for source, targets in source_to_targets.iteritems()
        }
        self.latent_rv_to_parents.update({
            label: set()
            for label in all_labels
            if label not in self.latent_rv_to_parents
        })

        self.latent_rv_to_children = defaultdict(lambda: set())
        for child, parents in self.latent_rv_to_parents.iteritems():
            for parent in parents:
                self.latent_rv_to_children[parent].add(child)
        self.latent_rv_to_children.update({
            label: set()
            for label in all_labels
            if label not in self.latent_rv_to_children
        })
        self.latent_rv_to_children = dict(self.latent_rv_to_children) 

        # Generate identifiers for each SVM score random variable
        self._svm_score_var_name = {
            label: "Evidence_" + pdoo.convert_node_to_str(label)
            for label in self.latent_rv_to_parents
        }

        # Map each latent random variable to its corresponding
        # SVM score random variable
        self.latent_rv_to_svm_score_var = {
            var: self._svm_score_var_name[var]
            for var in self.latent_rv_to_parents
        }

        svm_score_var_to_parents = {
            self.latent_rv_to_svm_score_var[var]: [var]
            for var in self.latent_rv_to_parents
        }
        svm_score_var_to_parents = {
            svm_score_var: list(sorted(parents))
            for svm_score_var, parents in svm_score_var_to_parents.iteritems()
        }

        svm_score_vars = set(svm_score_var_to_parents.keys())
        latent_label_vars = set(self.latent_rv_to_parents.keys())

        # Fit the conditional distributions of the SVM score random 
        # variables conditioned on each latent random variable
        item_to_index = {
            item: i
            for i,item in enumerate(training_items)
        }
        r = self._fit_score_distrs(
            training_feat_vecs,
            item_to_index,
            self.latent_rv_to_svm_score_var,
            item_to_group
        )
        self.svm_score_var_to_pos_distr = r[0]
        self.svm_score_var_to_neg_distr = r[1]
        self.svm_score_var_to_bin_bounds = r[2] 
    
   
    def _gibbs_iteration_naive(
            self,
            latent_rv_order,
            pos_rvs,
            skip_sampling,
            bayes_net_dag,
            node_to_descendents,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        ):
        """
        Sample from every node. This is the naive implementation.
        """
        from_pos_to_neg = set()
        from_neg_to_pos = set()
        for curr_rv in latent_rv_order:
            p_full_cond = _compute_full_conditional(
                curr_rv,
                self.trivial_labels,
                self.latent_rv_to_parents,
                self.latent_rv_to_children,
                self.latent_rv_to_svm_score_var,
                self.label_to_p_pos,                
                set(latent_rv_order) - pos_rvs,
                svm_var_to_pos_prob,
                svm_var_to_neg_prob
            )
            full_cond_sample = np.random.binomial(
                size=1,
                n=1,
                p=p_full_cond
            )[0]
            if DEBUG > 3:
                print "Sampled %d ~ %s" % (full_cond_sample, curr_rv)
            if full_cond_sample == 1 and curr_rv not in pos_rvs:
                from_neg_to_pos.add(curr_rv)
                pos_rvs.add(curr_rv)
            elif full_cond_sample == 0 and curr_rv in pos_rvs:
                from_pos_to_neg.add(curr_rv)
                pos_rvs.remove(curr_rv)
        return from_pos_to_neg, from_neg_to_pos

    def _gibbs_iteration_smart(
            self,
            latent_rv_order, 
            pos_rvs, 
            skip_sampling,
            bayes_net_dag,
            node_to_descendents,
            svm_var_to_pos_prob,
            svm_var_to_neg_prob
        ):
        """
        Perform shortcuts for not sampling from every node
        """
        from_pos_to_neg = set()
        from_neg_to_pos = set()
        for curr_rv in latent_rv_order:
            if curr_rv in skip_sampling:
                continue
            p_full_cond = _compute_full_conditional(
                curr_rv,
                self.trivial_labels,
                self.latent_rv_to_parents,
                self.latent_rv_to_children,
                self.latent_rv_to_svm_score_var,
                self.label_to_p_pos,
                set(latent_rv_order) - pos_rvs,
                svm_var_to_pos_prob,
                svm_var_to_neg_prob
            )
            full_cond_sample = np.random.binomial(
                size=1,
                n=1,
                p=p_full_cond
            )[0]
            _update_gibbs_sampling_state(
                curr_rv,
                full_cond_sample,
                pos_rvs,
                skip_sampling,
                from_pos_to_neg,
                from_neg_to_pos,
                bayes_net_dag
            )
        return from_pos_to_neg, from_neg_to_pos

    def predict(self, queries, verbose=False):
        label_to_scores = _compute_scores(
            queries, 
            self.label_to_classifier
        )
        r = _compute_evidence_cond_probs(
            label_to_scores,
            self._svm_score_var_name, 
            self.svm_score_var_to_bin_bounds,
            self.svm_score_var_to_pos_distr,
            self.svm_score_var_to_neg_distr
        ) 
        svm_var_to_pos_probs = r[0]
        svm_var_to_neg_probs = r[1]
        svm_var_to_scores = r[2]
        label_to_score_list = r[3]
        label_to_marginals = self._compute_marginals(
            svm_var_to_scores,
            svm_var_to_pos_probs,
            svm_var_to_neg_probs,
            verbose
        )        
        return label_to_marginals, label_to_score_list

    def _compute_marginals(
            self, 
            svm_var_to_scores, 
            svm_var_to_pos_probs, 
            svm_var_to_neg_probs,
            verbose=False,
            naive=False
        ):
        label_to_marginals = [] 
        label_to_marginal_sequence_list = [] 
        for q_i in range(len(list(svm_var_to_scores.values())[0])):
            svm_var_to_score = {
                svm_var: scores[q_i]
                for svm_var, scores in svm_var_to_scores.iteritems()
            }
            svm_var_to_pos_prob = {
                svm_var: probs[q_i]
                for svm_var, probs in svm_var_to_pos_probs.iteritems()
            }
            svm_var_to_neg_prob = {
                svm_var: probs[q_i]
                for svm_var, probs in svm_var_to_neg_probs.iteritems()
            }

            latent_var_to_count = defaultdict(lambda: 0)
            bayes_net_dag = DirectedAcyclicGraph(self.latent_rv_to_children)
            node_to_descendents = {
                node: bayes_net_dag.descendent_nodes(node) - set([node])
                for node in bayes_net_dag.get_all_nodes()
            }
            latent_rv_order = graph.topological_sort(bayes_net_dag)

            # Initialize the assignments of the latent random variables
            # to those that were predicted positive by each SVM
            pos_rvs = set()
            for rv in latent_rv_order:
                # The node was predicted positive
                svm_var = self.latent_rv_to_svm_score_var[rv]
                if not svm_var in svm_var_to_score:
                    pos_rvs.add(rv)
                    pos_rvs.update(node_to_descendents[rv])
                elif svm_var_to_score[svm_var] > 0.0:
                    pos_rvs.add(rv)
                    pos_rvs.update(node_to_descendents[rv])

            # Compute the samples that we can skip
            skip_sampling = set()
            for rv in pos_rvs:
                parents = self.latent_rv_to_parents[rv]
                if len(parents & pos_rvs) > 0:
                    skip_sampling.add(rv)
            for rv in set(latent_rv_order) - pos_rvs:
                children = self.latent_rv_to_children[rv]
                if len(children - pos_rvs) > 0:
                    skip_sampling.add(rv)

            # Run the burn-in
            print "running the Gibbs sampling burn-in..."
            for gibbs_iter in range(self.n_burn_in):
                if naive:
                    from_pos_to_neg, from_neg_to_pos = self._gibbs_iteration_naive(
                        latent_rv_order,
                        pos_rvs,
                        skip_sampling,
                        bayes_net_dag,
                        node_to_descendents,
                        svm_var_to_pos_prob,
                        svm_var_to_neg_prob
                    )
                else:
                    from_pos_to_neg, from_neg_to_pos = self._gibbs_iteration_smart(
                        latent_rv_order,
                        pos_rvs,
                        skip_sampling,
                        bayes_net_dag,
                        node_to_descendents,
                        svm_var_to_pos_prob,
                        svm_var_to_neg_prob
                    )
            print "done."

            # Initialize the tables used for computing counts that are 
            # necessary for estimating the marginal distributions
            rv_to_iter_change_neg_to_pos = {
                rv: 0
                for rv in pos_rvs
            }
            rv_to_count = {
                rv: 0
                for rv in latent_rv_order
            }
               

            #print 
            #print "---------------------------------------------------------"
            #print
            #print "STARTING THE GIBBS SAMPLER"
            #print "The current positive variables are:"
            #print pos_rvs 
            #print "Initialized mapping of random variable to last changed to positive:"
            #print rv_to_iter_change_neg_to_pos
            #print "Initialized mapping of random variable to count:"
            #print rv_to_count
     
            # Run the sampler
            print "running the Gibbs sampler..."
            if verbose:
                label_to_marginal_sequence = {
                    label: []
                    for label in latent_rv_order
                }
            for gibbs_iter in range(1, N_GIBBS_ITERS+1):
                #print "Gibbs sampling iteration %d" % gibbs_iter
                if naive:
                    from_pos_to_neg, from_neg_to_pos = self._gibbs_iteration_naive(
                        latent_rv_order,
                        pos_rvs,
                        skip_sampling,
                        bayes_net_dag,
                        node_to_descendents,
                        svm_var_to_pos_prob,
                        svm_var_to_neg_prob
                    )
                else:
                    from_pos_to_neg, from_neg_to_pos = self._gibbs_iteration_smart(
                        latent_rv_order,
                        pos_rvs,
                        skip_sampling,
                        bayes_net_dag,
                        node_to_descendents,
                        svm_var_to_pos_prob,
                        svm_var_to_neg_prob
                    )
                # Update the counts for those random variables that 
                # changed from positive to negative
                for rv in from_pos_to_neg:
                    rv_incr = gibbs_iter - rv_to_iter_change_neg_to_pos[rv]
                    #print "RV %s was last set to pos at iter %d. It is now iteration %d, so increment by %d" % (rv, rv_to_iter_change_neg_to_pos[rv], gibbs_iter, rv_incr)
                    rv_to_count[rv] += rv_incr
                    del rv_to_iter_change_neg_to_pos[rv]
                # Record the iteration for those random variables
                # that changed from negative to positive
                for rv in from_neg_to_pos:
                    rv_to_iter_change_neg_to_pos[rv] = gibbs_iter

                if DEBUG > 3:
                    print "Update pos to neg:"
                    print from_pos_to_neg
                    print "Update neg to pos:"
                    print from_neg_to_pos
                    print "The positive set is:"
                    print pos_rvs
                    print "Last udpate:"
                    print rv_to_iter_change_neg_to_pos
                    print "Counts:"
                    print rv_to_count
                    print

                if verbose:
                    for rv in latent_rv_order:
                        if rv in rv_to_iter_change_neg_to_pos:
                            add_count = gibbs_iter - rv_to_iter_change_neg_to_pos[rv]
                            label_to_marginal_sequence[rv].append(
                                (rv_to_count[rv] + add_count)/ float(gibbs_iter)
                            )    
                        else:
                            label_to_marginal_sequence[rv].append(
                                rv_to_count[rv] / float(gibbs_iter)
                            )

            if verbose:
                label_to_marginal_sequence_list.append(
                    label_to_marginal_sequence
                )       


            # Perform final update
            for rv, last_update in rv_to_iter_change_neg_to_pos.iteritems():
                rv_incr = gibbs_iter - last_update
                rv_to_count[rv] += rv_incr
            print "done."

            rv_to_marginal = {
                rv: float(rv_to_count[rv]) / N_GIBBS_ITERS
                for rv in latent_rv_order
            }
            label_to_marginals.append(rv_to_marginal)

        if self.artifact_dir and verbose:
            marginal_sequence_f = join(self.artifact_dir, "marginal_sequences.pickle")
            with open(marginal_sequence_f, 'w') as f:
                f.write(cPickle.dumps(label_to_marginal_sequence_list))
        else:
            print "Warning! Verbose setting is on, but no artifact location was \
                specified to store marginals sequence" 
        return label_to_marginals


class PerLabelSVM_BNC_DiscBins_StaticBins_Gibbs(PerLabelSVM_BNC_DiscBins_Gibbs):

    def __init__(
        self,
        binary_classifier_type,
        binary_classifier_params,
        assert_ambig_neg,
        pseudocount,
        prior_pos_estimation,
        n_bins_svm_score,
        n_burn_in,
        artifact_dir,
        delete_artifacts=True,
        downweight_by_group=False,
        constant_prior_pos=None
    ):
        """
        Args:
            prior_pos_estimation - method for which to estimate the 
                prior probability that a label is positive given
                all children are negative
        """
        super(PerLabelSVM_BNC_DiscBins_StaticBins_Gibbs, self).__init__(
            binary_classifier_type,
            binary_classifier_params,
            assert_ambig_neg,
            pseudocount,
            prior_pos_estimation,
            n_burn_in,
            artifact_dir,
            downweight_by_group=downweight_by_group,
            delete_artifacts=delete_artifacts,
            constant_prior_pos=constant_prior_pos
        )
        self.n_bins_svm_score = int(n_bins_svm_score)

    def _fit_score_distrs(
        self, 
        label_var_to_svm_score_var, 
        item_to_group=None
    ):
        svm_score_var_to_pos_distr = {}
        svm_score_var_to_neg_distr = {}
        svm_score_var_to_space = {}
        svm_score_var_to_bin_bounds = {}

        outcome_names = [
            "bin_%d" % bin_i
            for  bin_i in range(self.n_bins_svm_score)
        ]

        for label in self.label_to_pos_scores:
            pos_scores = self.label_to_pos_scores[label]
            neg_scores = self.label_to_neg_scores[label]
            all_scores = list(pos_scores) + list(neg_scores)
            all_scores = sorted(all_scores)

            bin_bounds = _even_space_bins(
                all_scores,
                self.n_bins_svm_score - 2
            )

            pos_probs = _compute_bin_probs(
                bin_bounds,
                pos_scores,
                self.pseudocount
            )

            neg_probs = _compute_bin_probs(
                bin_bounds,
                neg_scores,
                self.pseudocount
            )

            svm_score_var = label_var_to_svm_score_var[label]
            # NOTE! The order here is extremely important
            svm_score_var_to_pos_distr[svm_score_var] = pos_probs
            svm_score_var_to_neg_distr[svm_score_var] = neg_probs
            svm_score_var_to_space[svm_score_var] = outcome_names
            svm_score_var_to_bin_bounds[svm_score_var] = bin_bounds

        return (
            svm_score_var_to_pos_distr,
            svm_score_var_to_neg_distr,
            svm_score_var_to_bin_bounds
        )


def _partition_groups(item_to_group):
    """
    Partition the groups into two groups in an attempt
    to ensure that the partitions are even-sized. This
    is the knapsack problem, and we solve it here using
    hill-climbing.

    TODO: Since this is integer valued knapsacking, this
    can be solved with dynamic programming, which I 
    should implement.
    """
    def _obj_fun(
            group_to_incl, 
            group_to_items
        ):
        n_total = sum([
            len(items) 
            for items in group_to_items.values()
        ])
        n_included = sum([
            len(items) 
            for group, items  in group_to_items.iteritems()
            if group_to_incl[group]
        ])
        frac = float(n_included) / n_total
        diff_half = abs(frac - 0.5)
        return diff_half

    def _hill_climb(
            group_to_incl, 
            group_to_items
        ):
        curr_obj = _obj_fun(
            group_to_incl, 
            group_to_items
        )
        obj_improved = True
        while obj_improved:
            obj_improved = False
            for group in group_to_incl:
                group_to_incl[group] = not group_to_incl[group]
                obj = _obj_fun(
                    group_to_incl, 
                    group_to_items
                )
                if obj < curr_obj:
                    obj_improved = True
                    curr_obj = obj
                else:
                    group_to_incl[group] = not group_to_incl[group]
        return curr_obj, group_to_incl

    group_to_items = defaultdict(lambda: [])
    for item, group in item_to_group.iteritems():
        group_to_items[group].append(item)
    group_to_items = dict(group_to_items)

    groups = set(item_to_group.values())
    min_group_to_included = None
    min_obj = 1.0

    iters = 10
    #random.seed(88)
    for iter_i in range(iters):
        # Randomly assign each group to be included
        # or not to be included
        group_to_incl = {
            group: bool(random.randint(0, 1))
            for group in groups
        }
        obj, group_to_incl = _hill_climb(
            group_to_incl, 
            group_to_items
        )
        if obj < min_obj:
            min_group_to_included = {
                k:v
                for k,v in group_to_incl.iteritems()
            }        
            min_obj = obj
    return min_group_to_included


def _histogram_estimator_pdf(bin_bounds, densities, x):
    """
    Compute the log-density for a histogram density
    estimator.

    Args:
        bin_bounds: the bin boundaries. For example:
                [0.0, 0.75, 3.2]
            defines 2 bins:
                [0.0 0.75), [0.75, 3.2]
        densities: the density for each corresponding
            bin
    """    
    for bin_i in range(len(bin_bounds)-1):
        left = bin_bounds[bin_i]
        right = bin_bounds[bin_i + 1]
        if x >= left and x < right:
            return math.log(densities[bin_i])


def _likelihood_for_fold(
        bin_bounds,  
        train_scores, 
        test_scores, 
        pseudo_mass=1.0
    ):
    pseudo = pseudo_mass / (len(bin_bounds)-1.0)
    counts = [
        pseudo
        for i in range(len(bin_bounds)-1)
    ]
    for score in train_scores:
        for bin_i in range(len(bin_bounds)-1):
            left = bin_bounds[bin_i]
            right = bin_bounds[bin_i + 1]
            if score >= left and score < right:
                counts[bin_i] += 1.0
                break

    bin_width = bin_bounds[1] - bin_bounds[0]

    total_count = sum(counts)
    densities = [
        count / (bin_width * total_count)
        for count in counts
    ]

    log_like = 0.0
    for score in test_scores:
        log_like += _histogram_estimator_pdf(
            bin_bounds,
            densities,
            score
        )
    return log_like


def _avg_likelihood_for_bin_size(
        all_scores,
        fold_1,
        fold_2,
        n_bins
    ):
    bin_bounds =_even_space_bins(
        all_scores,
        n_bins
    )
    like_1 = _likelihood_for_fold(bin_bounds, fold_1, fold_2)
    like_2 = _likelihood_for_fold(bin_bounds, fold_2, fold_1)
    return (like_1 + like_2) / 2.0
   

def _bin_estimation(
        pos_items,
        neg_items,
        pos_scores,
        neg_scores,
        item_to_group,
        default_bins
    ):
    """
    Estimate the number of bins using a 2-fold cross
    validation in an attempt to optimize the average
    likelihood of the folds based on histogram density
    estimator. The overall procedure works as follows:
        1. Set the range of the histogram using both
            positive and negative examples.
        2. Use cross validation to estimate the number 
            of bins for a histogram density estimate by 
            optimizing the likelihood of the held out 
            data using only the positive examples.
    """
    all_scores = list(pos_scores) + list(neg_scores)
    if len(all_scores) == 1:
        return all_scores

    item_to_score = {
        pos_items[i]: pos_scores[i]
        for i in range(len(pos_scores))
    }
    pos_item_to_group = {
        item: item_to_group[item]
        for item in pos_items
    }

    if len(pos_items) == 1:
        # If there is only one positive item, then simply use a 
        # default number of bins. This is a degenerate case, in which
        # learning will likely be impossible anyway.
        return _even_space_bins(
            all_scores,
            default_bins
        )
    elif len(set(pos_item_to_group.values())) > 1:
        group_to_incl = _partition_groups(
            pos_item_to_group
        )
        fold_1 = [
            item_to_score[item]
            for item in pos_items
            if group_to_incl[
                item_to_group[item]
            ]
        ]
        fold_2 = [
            item_to_score[item]
            for item in pos_items
            if not group_to_incl[
                item_to_group[item]
            ]
        ]
    else:
        return _even_space_bins(
            all_scores,
            default_bins
        )
    # Find the best number of bins using a grid-search
    n_bins_best = 0
    like_best = float('-inf')
    n_bin_cands = [x for x in range(1,10)]
    n_bin_cands += [x for x in range(10,30,2)]
    n_bin_cands += [x for x in range(30,100,5)]
    n_bin_cands += [x for x in range(100,200,10)]
    for n_bins in n_bin_cands:
        like = _avg_likelihood_for_bin_size(
            all_scores,
            fold_1,
            fold_2,
            n_bins
        )
        if like > like_best:
            n_bins_best = n_bins
            like_best = like
    bin_bounds =_even_space_bins(
        all_scores,
        n_bins_best
    )
    return bin_bounds








 

#def _bin_estimation(
#        pos_scores_fold_1,
#        neg_scores_fold_1
#        pos_scores_fold_2,
#        neg_scores_fold_2
#        default_bins
#    ):
#    """
#    Estimate the number of bins using a 2-fold cross
#    validation in an attempt to optimize the average
#    likelihood of the folds based on histogram density
#    estimator. The overall procedure works as follows:
#        1. Set the range of the histogram using both
#            positive and negative examples.
#        2. Use cross validation to estimate the number 
#            of bins for a histogram density estimate by 
#            optimizing the likelihood of the held out 
#            data using only the positive examples.
#    """
#    all_scores = list(pos_scores_fold_1) + list(neg_scores_fold_1) \
#        + list(pos_scores_fold_2) + list(neg_scores_fold_2)
#    if len(all_scores) == 1:
#        return all_scores
#
#    if len(pos_scores_fold_1) == 0 or len(pos_scores_fold_2) == 0:
#        # If we don't have twfolds worth of data then simply use a 
#        # default number of bins. This is a degenerate case, in which
#        # learning will likely be impossible anyway.
#        # TODO Is this right???
#        return _even_space_bins(
#            all_scores,
#            default_bins
#        )
#   
#    fold_1 = list(pos_scores_fold_1) 
#    fold_2 = list(pos_scores_fold_2)
# 
#    # Find the best number of bins using a grid-search
#    n_bins_best = 0
#    like_best = float('-inf')
#    n_bin_cands = [x for x in range(1,10)]
#    n_bin_cands += [x for x in range(10,30,2)]
#    n_bin_cands += [x for x in range(30,100,5)]
#    n_bin_cands += [x for x in range(100,200,10)]
#    for n_bins in n_bin_cands:
#        like = _avg_likelihood_for_bin_size(
#            all_scores,
#            fold_1,
#            fold_2,
#            n_bins
#        )
#        if like > like_best:
#            n_bins_best = n_bins
#            like_best = like
#    bin_bounds =_even_space_bins(
#        all_scores,
#        n_bins_best
#    )
#    return bin_bounds






def _compute_folds(
        pos_items,
        neg_items,
        item_to_group
    ):  
    pos_item_to_group = {
        item: item_to_group[item]
        for item in pos_items
    }
    pos_group_to_incl = _partition_groups(
        pos_item_to_group
    )
    fold_1_pos = [
        item
        for item in pos_items
        if pos_group_to_incl[
            item_to_group[item]
        ]
    ]
    fold_2_pos = [
        item
        for item in pos_items
        if not pos_group_to_incl[
            item_to_group[item]
        ]
    ]

    # Not every negative item is necessarily assigned
    # to a fold, so we need to assign these unassiged
    # items
    neg_items_unassigned = [
        item
        for item in neg_items
        if item_to_group[item] not in pos_group_to_incl
    ]
    unnasigned_neg_item_to_group = {
        item: item_to_group[item]
        for item in neg_items_unassigned
    }
    neg_group_to_incl = _partition_groups(
        unnasigned_neg_item_to_group
    )
    fold_1_neg = [
        item
        for item in neg_items
        if item_to_group[item] in pos_group_to_incl
        and pos_group_to_incl[
            item_to_group[item]
        ]
    ]
    fold_1_neg += [
        item
        for item in neg_items_unassigned
        if neg_group_to_incl[
            item_to_group[item]
        ]
    ]
    fold_2_neg = [
        item
        for item in neg_items
        if item_to_group[item] in pos_group_to_incl
        and not pos_group_to_incl[
            item_to_group[item]
        ]
    ]
    fold_2_neg += [
        item
        for item in neg_items_unassigned
        if not neg_group_to_incl[
            item_to_group[item]
        ]
    ]

    # Make sure there are no duplicates. If this
    # happened something went wrong
    assert len(set(fold_1_neg)) == len(fold_1_neg)
    assert len(set(fold_2_neg)) == len(fold_2_neg)

    item_to_group_fold_1 = {
        item: item_to_group[item]
        for item in fold_1_pos
    }
    item_to_group_fold_1.update({
        item: item_to_group[item]
        for item in fold_1_neg
    })

    item_to_group_fold_2 = {
        item: item_to_group[item]
        for item in fold_2_pos
    }
    item_to_group_fold_2.update({
        item: item_to_group[item]
        for item in fold_2_neg
    })

    return (
        fold_1_pos,
        fold_1_neg,
        fold_2_pos,
        fold_2_neg,
        item_to_group_fold_1,
        item_to_group_fold_2
    )


def _compute_cross_fold_val_scores(
        fold_1_pos,
        fold_1_neg,
        fold_2_pos,
        fold_2_neg,
        item_to_group_fold_1,
        item_to_group_fold_2,
        training_feat_vecs,
        item_to_index,
        binary_classifier_type,
        binary_classifier_params,
        downweight_by_group
    ):
    """
    Estimate the number of bins using a 2-fold cross
    validation in an attempt to optimize the average
    likelihood of the folds based on histogram density
    estimator. The overall procedure works as follows:
        1. Set the range of the histogram using both
            positive and negative examples.
        2. Use cross validation to estimate the number 
            of bins for a histogram density estimate by 
            optimizing the likelihood of the held out 
            data using only the positive examples.
    """
    print "Running 2-fold cross-validation to obtain score estimates..."
 
    # Compute the feature vectors and class labels for 
    # the first fold
    pos_feat_vecs_fold_1 = [
        training_feat_vecs[
            item_to_index[x]
        ]
        for x in fold_1_pos
    ]
    neg_feat_vecs_fold_1 = [
        training_feat_vecs[
            item_to_index[x]
        ]
        for x in fold_1_neg
    ]
    feat_vecs_fold_1 = np.asarray(
        pos_feat_vecs_fold_1 + neg_feat_vecs_fold_1
    )
    pos_classes_fold_1 = [1 for x in fold_1_pos]
    neg_classes_fold_1 = [-1 for x in fold_1_neg]
    classes_fold_1 = np.asarray(
        pos_classes_fold_1 + neg_classes_fold_1
    )

    # Compute the feature vectors and class labels for 
    # the second fold
    pos_feat_vecs_fold_2 = [
        training_feat_vecs[
            item_to_index[x]
        ]
        for x in fold_2_pos
    ]
    neg_feat_vecs_fold_2 = [
        training_feat_vecs[
            item_to_index[x]
        ]
        for x in fold_2_neg
    ]
    feat_vecs_fold_2 = np.asarray(
        pos_feat_vecs_fold_2 + neg_feat_vecs_fold_2
    )
    pos_classes_fold_2 = [1 for x in fold_2_pos]
    neg_classes_fold_2 = [-1 for x in fold_2_neg]
    classes_fold_2 = np.asarray(
        pos_classes_fold_2 + neg_classes_fold_2
    )

    fold_1 = fold_1_pos + fold_1_neg
    fold_2 = fold_2_pos + fold_2_neg

    # Build the models
    model_fold_1 = bc.build_binary_classifier(
        binary_classifier_type,
        binary_classifier_params
    )
    model_fold_2 = bc.build_binary_classifier(
        binary_classifier_type,
        binary_classifier_params
    )

    # Fit the models
    if len(pos_feat_vecs_fold_1) > 0 and len(frozenset(classes_fold_1)) > 1:
        print "Training model on the first fold..."
        model_fold_1.fit(
            fold_1,
            feat_vecs_fold_1,
            classes_fold_1,
            item_to_group_fold_1,
            downweight_by_group=downweight_by_group
        )
        print "done."
        
        # Apply the model trained on fold 1 to fold 2
        pos_index = 0
        for index, clss in enumerate(model_fold_1.classes_):
            if clss == 1:
                pos_index = index
                break
        if len(fold_2_pos) > 0:
            pos_scores_fold_2 = [
                x
                for x in model_fold_1.decision_function(pos_feat_vecs_fold_2)
            ]
        else:
            pos_scores_fold_2 = []
        if len(fold_2_neg) > 0:
            neg_scores_fold_2 = [
                x
                for x in model_fold_1.decision_function(neg_feat_vecs_fold_2)
            ]
        else:
            neg_scores_fold_2 = []
    else:
        pos_scores_fold_2 = []
        neg_scores_fold_2 = []

    if len(pos_feat_vecs_fold_2) > 0 and len(frozenset(classes_fold_2)) > 1:
        print "Training model on the second fold..."
        model_fold_2.fit(
            fold_2,
            feat_vecs_fold_2,
            classes_fold_2,
            item_to_group_fold_2,
            downweight_by_group=downweight_by_group
        )
        print "done."

        # Apply the model trained on fold 2 to fold 1
        pos_index = 0
        for index, clss in enumerate(model_fold_2.classes_):
            if clss == 1:
                pos_index = index
                break

        if len(fold_1_pos) > 0:
            pos_scores_fold_1 = [
                x
                for x in model_fold_2.decision_function(pos_feat_vecs_fold_1)
            ]
        else:
            pos_scores_fold_1 = []
        if len(fold_1_neg) > 0:
            neg_scores_fold_1 = [
                x
                for x in model_fold_2.decision_function(neg_feat_vecs_fold_1)
            ]
        else:
            neg_scores_fold_1 = []
    else:
        pos_scores_fold_1 = []
        neg_scores_fold_1 = []

    return (
        pos_scores_fold_1,
        neg_scores_fold_1,
        pos_scores_fold_2,
        neg_scores_fold_2
    )

 
class PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs(PerLabelSVM_BNC_DiscBins_Gibbs):

    def __init__(
            self,
            binary_classifier_type,
            binary_classifier_params,
            assert_ambig_neg,
            pseudocount,
            prior_pos_estimation,
            n_burn_in,
            artifact_dir,
            n_bins_default,
            delete_artifacts=True,
            downweight_by_group=False,
            constant_prior_pos=None
        ):
        """
        Args:
            prior_pos_estimation - method for which to estimate the 
                prior probability that a label is positive given
                all children are negative
        """
        super(PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs, self).__init__(
            binary_classifier_type,
            binary_classifier_params,
            assert_ambig_neg,
            pseudocount,
            prior_pos_estimation,
            n_burn_in,
            artifact_dir,
            delete_artifacts=delete_artifacts,
            downweight_by_group=downweight_by_group,
            constant_prior_pos=constant_prior_pos
        )
        self.n_bins_default = n_bins_default

    def _fit_score_distrs(
            self,
            train_vecs,
            item_to_index,
            label_var_to_svm_score_var,
            item_to_group=None
        ):
        svm_score_var_to_pos_distr = {}
        svm_score_var_to_neg_distr = {}
        svm_score_var_to_space = {}
        svm_score_var_to_bin_bounds = {}
        for label in self.label_to_pos_scores:
            pos_items = self.label_to_pos_items[label]
            neg_items = self.label_to_neg_items[label]
            
            r = _compute_folds(
                pos_items,
                neg_items,
                item_to_group
            )
            fold_1_pos = r[0]
            fold_1_neg = r[1]
            fold_2_pos = r[2]
            fold_2_neg = r[3]
            item_to_group_fold_1 = r[4]
            item_to_group_fold_2 = r[5]

            r = _compute_cross_fold_val_scores(
                fold_1_pos,
                fold_1_neg,
                fold_2_pos,
                fold_2_neg,
                item_to_group_fold_1,
                item_to_group_fold_2,
                train_vecs,
                item_to_index,
                self.binary_classifier_type,
                self.binary_classifier_params,
                self.downweight_by_group
            )
            pos_scores_fold_1 = r[0]
            neg_scores_fold_1 = r[1]
            pos_scores_fold_2 = r[2]
            neg_scores_fold_2 = r[3]

            
            pos_scores = list(pos_scores_fold_1) + list(pos_scores_fold_2)
            neg_scores = list(neg_scores_fold_1) + list(neg_scores_fold_2)
             
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                # If we were unable to estimate scores using
                # 2-fold cross-valdiation, then fall back on
                # using the training scores
                print "Falling back on training scores..."
                pos_scores = self.label_to_pos_scores[label]
                neg_scores = self.label_to_neg_scores[label]
                bin_bounds = _bin_estimation(
                    pos_items,
                    neg_items,
                    pos_scores,
                    neg_scores,
                    item_to_group,
                    self.n_bins_default
                )
            else:
                bin_bounds = _bin_estimation(
                    fold_1_pos + fold_2_pos,
                    fold_1_neg + fold_2_neg,
                    pos_scores,
                    neg_scores,
                    item_to_group,
                    self.n_bins_default
                )
            pos_probs = _compute_bin_probs(
                bin_bounds,
                pos_scores,
                self.pseudocount
            )
            neg_probs = _compute_bin_probs(
                bin_bounds,
                neg_scores,
                self.pseudocount
            )
            outcome_names = [
                "bin_%d" % bin_i
                for  bin_i in range(len(pos_probs))
            ]
            svm_score_var = self.latent_rv_to_svm_score_var[label]
            svm_score_var_to_pos_distr[svm_score_var] = pos_probs
            svm_score_var_to_neg_distr[svm_score_var] = neg_probs
            svm_score_var_to_bin_bounds[svm_score_var] = bin_bounds
        return (
            svm_score_var_to_pos_distr,     
            svm_score_var_to_neg_distr,
            svm_score_var_to_bin_bounds
        )

    def fit(
            self,
            training_feat_vecs,
            training_items,
            item_to_labels,
            label_graph,
            item_to_group=None,
            verbose=False,
            feat_names=None
        ):
        assert item_to_group != None
        super(PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs, self).fit(
            training_feat_vecs,
            training_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            feat_names=feat_names
        )

 
def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
