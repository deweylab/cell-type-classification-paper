###################################################################################
#   Methods for loading and normalizing expression-based features to be used
#   for machine learning applications.
###################################################################################

import sys
from optparse import OptionParser
import numpy as np
from collections import namedtuple
import pkg_resources as pr
import json

sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/recount2")
sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/my_kallisto_pipeline")

import recount_data_manager as rdm
import kallisto_quantified_data_manager_hdf5 as kqdm

resource_package = __name__
Entry = namedtuple("Entry", ["gene", "log2fold", "padj"])

def _log_cpm(data_matrix):
    print "normalizing..."
    data_matrix = np.asarray([
        x/sum(x)
        for x in data_matrix
    ])
    data_matrix *= 1000000
    print "done."
    print "computing log of data..."
    data_matrix = np.log(data_matrix + 1)
    print "done."
    return data_matrix


def log_recount_gene_counts(
        sample_accs, 
        assert_all_data_retrieved=True
    ):    
    """
    log-CPM features from the recount2 data repository
    
    Args:
        sample_accs: a list of samples for which we are to
            retrieve a log-CPM expression vector
        assert_all_data_retrieved: assert that we have 
            retrieved data for all samples in sample_accs
    Returns:
        returned_sample_accs: list of sample accessions
            for which data was retrieved
        data_matrix: a DxG matrix of the data where D is
            the length of returned_sample_accs corresponding
            to each returned sample and G is the number of
            features. The rows are ordered according to 
            returned_sample_accs.
    """
    returned_sample_accs, data_matrix = recount_gene_counts(
        sample_accs,
        assert_all_data_retrieved=assert_all_data_retrieved
    )
    data_matrix = _log_cpm(data_matrix)
    return returned_sample_accs, data_matrix


def log_kallisto_gene_counts(
        sample_accs,
        assert_all_data_retrieved=True
    ):
    """
    log-CPM features from my Kallisto-quantified data
    """
    returned_sample_accs, returned_exp_accs, data_matrix, gene_names = kallisto_gene_counts(
        sample_accs,
        assert_all_data_retrieved=assert_all_data_retrieved,
    )
    data_matrix = _log_cpm(data_matrix)
    return returned_sample_accs, returned_exp_accs, data_matrix, gene_names    


def log_kallisto_gene_counts_for_experiments(
        exp_accs,
        assert_all_data_retrieved=True
    ):
    """
    log-CPM features from my Kallisto-quantified data
    """
    returned_exp_accs, data_matrix, gene_names = kallisto_gene_counts_for_experiments(
        exp_accs,
        assert_all_data_retrieved=assert_all_data_retrieved,
    )
    data_matrix = _log_cpm(data_matrix)
    return returned_exp_accs, data_matrix, gene_names


def log_cpm_kallisto_genes_for_experiments_log2fold_5_0_padj_0_05(
        exp_accs,
        assert_all_data_retrieved=True
    ):
    log2fold_thresh = 5.0
    padj_thresh = 0.05
    return _log_cpm_kallisto_genes_for_experiments_thresh_diff_polyA_vs_total(
        exp_accs,
        log2fold_thresh,
        padj_thresh,
        assert_all_data_retrieved
    )


def log_cpm_kallisto_genes_for_experiments_log2fold_2_5_padj_0_05(
        exp_accs,
        assert_all_data_retrieved=True
    ):
    log2fold_thresh = 2.5
    padj_thresh = 0.05
    return _log_cpm_kallisto_genes_for_experiments_thresh_diff_polyA_vs_total(
        exp_accs,
        log2fold_thresh,
        padj_thresh,
        assert_all_data_retrieved
    )

def _log_cpm_kallisto_genes_for_experiments_thresh_diff_polyA_vs_total(
        exp_accs,
        log2fold_thresh,
        padj_thresh,
        assert_all_data_retrieved=True
    ):
    gene_poly_vs_total_entries = _load_polya_vs_total_features()
    remove_genes = set([
        entry.gene 
        for entry in gene_poly_vs_total_entries
        if entry.log2fold < 0                       # we only want to filter genes for which PolyA decreased gene level 
        and abs(entry.log2fold) > log2fold_thresh   
        and entry.padj < padj_thresh
    ])
    print "%d genes removed..." % len(remove_genes)

    returned_exp_accs, data_matrix, gene_names = kallisto_gene_counts_for_experiments(
        exp_accs,
        assert_all_data_retrieved=assert_all_data_retrieved,
        remove_genes=remove_genes
    ) 
    data_matrix = _log_cpm(data_matrix)
    return returned_exp_accs, data_matrix, gene_names


def kallisto_tpm_for_experiments(
        exp_accs,
        assert_all_data_retrieved=True
    ):
    """
    TPM features from my Kallisto-quantified data
    """
    print "loading TPM's data...."
    returned_exp_accs, data_matrix, gene_names = kqdm.get_gene_tpms_for_experiments(
        exp_accs
    )
    if assert_all_data_retrieved: 
        assert frozenset(returned_exp_accs) == frozenset(exp_accs)
    print "done."
    return returned_exp_accs, data_matrix, gene_names


def kallisto_binary_threshhold_tpm_0_1(exp_accs, assert_all_data_retrieved=True):
    return kallisto_binary_threshhold_tpm(exp_accs, 0.1)


def kallisto_binary_threshhold_tpm_1_0(exp_accs, assert_all_data_retrieved=True):
    return kallisto_binary_threshhold_tpm(exp_accs, 1.0)


def kallisto_binary_threshhold_tpm_10_0(exp_accs, assert_all_data_retrieved=True):
    return kallisto_binary_threshhold_tpm(exp_accs, 10.0)


def kallisto_binary_threshhold_tpm_100_0(exp_accs, assert_all_data_retrieved=True):
    return kallisto_binary_threshhold_tpm(exp_accs, 100.0)


def kallisto_binary_threshhold_tpm(exp_accs, threshold, assert_all_data_retrieved=True):
    returned_exp_accs, data_matrix, gene_names = kallisto_tpm_for_experiments(
        exp_accs,
        assert_all_data_retrieved=assert_all_data_retrieved
    )
    data_matrix = np.where(data_matrix > threshold, 1, 0)
    return returned_exp_accs, data_matrix, gene_names 


def recount_gene_counts(
        sample_accs, 
        assert_all_data_retrieved=True
    ):
    """
    Raw counts from the recount2 data repository
    """
    print "loading recount data..."
    returned_sample_accs, data_matrix = rdm.get_gene_counts_for_samples(
        sample_accs,
        one_experiment_per_sample=True 
    )
    if assert_all_data_retrieved:
        assert frozenset(returned_sample_accs) == frozenset(sample_accs)
    print "done."
    return returned_sample_accs, data_matrix 


def kallisto_gene_counts_for_experiments(
        experiment_accs,
        assert_all_data_retrieved=True,
        remove_genes=None
    ):
    """
    Expected counts for each gene from my Kallisto-quantified data
    """
    print "loading counts data..."
    returned_exp_accs, data_matrix, gene_names = kqdm.get_gene_counts_for_experiments(
        experiment_accs,
        remove_genes=remove_genes
    )
    if assert_all_data_retrieved:
        assert frozenset(returned_exp_accs) == frozenset(experiment_accs)
    print "done."
    return returned_exp_accs, data_matrix, gene_names


def log_kallisto_marker_gene_counts_for_experiments(
        experiment_accs,
        assert_all_data_retrieved=True
    ):
    marker_gene_f = pr.resource_filename(resource_package, "cellmarker_cell_type_to_ensembl_id_markers.json")
    with open(marker_gene_f, 'r') as f:
        cell_type_to_marker_info = json.load(f)
    marker_genes = set()
    for cell_type, marker_info in cell_type_to_marker_info.iteritems():
        #print "Cell type: %s" % cell_type
        #print "%d pub marker datas" % len(marker_info)
        for pub_marker_data in marker_info:
            #print "%d gene_sets" % len(pub_marker_data['marker_gene_or_sets'])
            for gene_set in pub_marker_data['marker_gene_or_sets']:
                #print "%d or sets" % len(gene_set)
                for or_set in gene_set:
                    if isinstance(or_set, basestring):
                        continue
                    marker_genes.update(or_set)
    print "There are %d total marker genes" % len(marker_genes) 
    genes_in_db = set(kqdm.get_all_gene_names_in_hg38_v24_kallisto())
    remove_genes = genes_in_db - marker_genes
    print "Number of genes not considering because they are not markers: %d" % len(remove_genes)
    returned_exp_accs, data_matrix, gene_names = kallisto_gene_counts_for_experiments(
        experiment_accs,
        assert_all_data_retrieved=assert_all_data_retrieved,
        remove_genes = remove_genes
    )
    print "Total gene features used: %d" % len(gene_names)
    return returned_exp_accs, data_matrix, gene_names
     

def kallisto_isoform_counts_for_experiments(
        experiment_accs,
        assert_all_data_retrieved=True,
        remove_genes=None
    ):
    print "loading counts data..."
    returned_exp_accs, data_matrix = kqdm.get_transcript_counts_for_experiments(
        experiment_accs
    )
    if assert_all_data_retrieved:
        assert frozenset(returned_exp_accs) == frozenset(experiment_accs)
    print "done."
    return returned_exp_accs, data_matrix, None # TODO we should retrieve the transcript names


def kallisto_gene_counts(
        sample_accs,
        assert_all_data_retrieved=True
    ):
    """
    Raw expected-counts from my Kallisto-quantified data
    """
    print "loading counts data..."
    returned_sample_accs, returned_exp_accs, data_matrix, gene_names = kqdm.get_gene_counts_for_samples(
        sample_accs,
        one_experiment_per_sample=True
    )
    if assert_all_data_retrieved:
        assert frozenset(returned_sample_accs) == frozenset(sample_accs)
    print "done."
    return returned_sample_accs, returned_exp_accs, data_matrix, gene_names


def _load_polya_vs_total_features():
    # "","row","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj"
    entries = []
    genes_no_padj = set()
    genes_no_log2fold = set()
    poly_vs_total_f = pr.resource_filename(resource_package, "polya_vs_total.csv")
    with open(poly_vs_total_f, 'r') as f:
        for i,l in enumerate(f):
            if i == 0:
                continue
            toks = l.split(',')
            gene = toks[1][1:-1]
            try:
                log2fold = float(toks[3])
            except:
                print "Unable to parse log2fold for gene %s. Raw token: %s" % (
                    gene,
                    toks[3]
                )
                log2fold = None
                genes_no_log2fold.add(gene)
            try:
                padj = float(toks[7])
            except:
                print "Unable to parse padj for gene %s. Raw token: %s" % (
                    gene, 
                    toks[7]
                )
                padj = None
                genes_no_padj.add(gene)
            entry = Entry(gene, log2fold, padj)
            if gene and log2fold and padj:
                entries.append(entry)
    sorted_by_logfold = sorted(entries, key=lambda x: abs(x.log2fold), reverse=True)
    #print sorted_by_logfold[0:1000]
    print "%d entries have no log2fold" % len(genes_no_log2fold)
    print "%d entries have no pajd" % len(genes_no_padj)
    print "%d genes with either no log2fold or no padj" % len(genes_no_log2fold | genes_no_padj)
    print "%d total entries" % len(entries)
    return sorted_by_logfold


def random_features(
        experiment_accs,
        assert_all_data_retrieved=True,
        remove_genes=None
    ):
    return experiment_accs, np.random.randn(len(experiment_accs), 10), range(10)

def featurize(
        featurizer_name, 
        experiment_accs, 
        assert_all_data_retrieved=True
    ):
    name_to_featurizer = {
        'random': random_features,
        'log_recount_gene_counts': log_recount_gene_counts,
        'recount_gene_counts': recount_gene_counts,
        'kallisto_gene_counts': kallisto_gene_counts_for_experiments,
        'kallisto_isoform_counts': kallisto_isoform_counts_for_experiments,
        'kallisto_gene_log_cpm': log_kallisto_gene_counts_for_experiments,
        'kallisto_gene_tpm': kallisto_tpm_for_experiments,
        'kallisto_gene_log_cpm_logfold_5.0_padj_0.05': log_cpm_kallisto_genes_for_experiments_log2fold_5_0_padj_0_05,
        'kallisto_gene_log_cpm_logfold_2.5_padj_0.05': log_cpm_kallisto_genes_for_experiments_log2fold_2_5_padj_0_05,
        'kallisto_binary_threshhold_tpm_0.1': kallisto_binary_threshhold_tpm_0_1,
        'kallisto_binary_threshhold_tpm_1.0': kallisto_binary_threshhold_tpm_1_0,
        'kallisto_binary_threshhold_tpm_10.0': kallisto_binary_threshhold_tpm_10_0,
        'kallisto_binary_threshhold_tpm_100.0': kallisto_binary_threshhold_tpm_100_0,
        'kalliost_marker_gene_log_cpm': log_kallisto_marker_gene_counts_for_experiments
    }
    assert featurizer_name in name_to_featurizer
    return name_to_featurizer[featurizer_name](
        experiment_accs, 
        assert_all_data_retrieved=assert_all_data_retrieved
    )


def main():

    #returned_exp_accs, data_matrix, gene_names = log_cpm_kallisto_genes_for_experiments_log2fold_2_5_padj_0_05(["SRX2513454"])
    #returned_exp_accs, data_matrix, gene_names = log_kallisto_marker_gene_counts_for_experiments(["SRX2513454"])
    returned_exp_accs, data_matrix, gene_names = log_kallisto_gene_counts_for_experiments(["SRX2513454"])
    gene_names = set(gene_names)
    target_genes = set([
        "ENSG00000121594",
        "ENSG00000114013",
        "ENSG00000103855",
        "ENSG00000134258",
        "ENSG00000120217",
        "ENSG00000197646",
        "ENSG00000160223"
    ])

    assert target_genes < gene_names
    print "HERE!"

    #print list(data_matrix[0])
    #r = kallisto_gene_counts_for_experiments(["SRX2513454"])
    #print len(r[2])
    #_load_polya_vs_total_features()

if __name__ == "__main__":
    main()
