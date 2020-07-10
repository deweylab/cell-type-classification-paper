##################################################################################
#   For a given analysis, we often need a bunch of information in regards to 
#   the experiments, the cell type label graph, the gene expression data. This 
#   function computes all of this 'boiler plate' data.
##################################################################################

from optparse import OptionParser
import sys
import os
from os.path import join
import json
import collections
from collections import defaultdict
import h5py
import numpy as np
import pkg_resources as pr

resource_package = __name__

import kallisto_quantified_data_manager_hdf5_py3 as kqdm

def main():
    usage = ""
    parser = OptionParser()
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    exp_set_f = args[0]
    features = args[1]
    out_f = options.out_file 
    _build_dataset(features, exp_set_f, out_f)


def _retrieve_tpm(exp_accs):
    """
    TPM features from my Kallisto-quantified data
    """
    print("loading TPM for {} experiments...".format(len(exp_accs)))
    returned_exp_accs, data_matrix, gene_ids = kqdm.get_gene_tpms_for_experiments(
        exp_accs
    )
    assert frozenset(returned_exp_accs) == frozenset(exp_accs)
    print("done.")
    return returned_exp_accs, data_matrix, gene_ids


def _retrieve_counts(exp_accs):
    """
    Expected counts features from my Kallisto-quantified data
    """
    print("loading CPM for {} experiments...".format(len(exp_accs)))
    returned_exp_accs, data_matrix, gene_ids = kqdm.get_gene_counts_for_experiments(
        exp_accs
    )
    assert frozenset(returned_exp_accs) == frozenset(exp_accs)
    print("done.")
    return returned_exp_accs, data_matrix, gene_ids


def _retrieve_log_tpm(exp_accs):
    """
    log-TPM features from my Kallisto-quantified data
    """
    exp_accs, data_matrix, gene_ids = _retrieve_tpm(exp_accs)
    print('converting to log(TPM+1)...')
    data_matrix = np.log(data_matrix + 1)
    print('done.')
    return exp_accs, data_matrix, gene_ids


def _retrieve_log_cpm(exp_accs):
    """
    log-CPM features from my Kallisto-quantified data
    """
    exp_accs, data_matrix, gene_ids = _retrieve_counts(exp_accs)
    print('converting to log(CPM+1)...')
    data_matrix = np.array([
        x/sum(x)
        for x in data_matrix
    ])
    data_matrix *= 1e6
    data_matrix = np.log(data_matrix + 1)
    print('done.')
    return exp_accs, data_matrix, gene_ids


def _retrieve_log_tpm_10x_genes(exp_accs):
    """
    log-TPM features from my Kallisto-quantified data 
    for only genes that are also in the 10x datasets.
    """
    tenx_genes_f = pr.resource_filename(resource_package, "10x_genes.json")
    with open(tenx_genes_f, 'r') as f:
        tenx_genes = set(json.load(f))
    all_genes = set(kqdm.get_all_gene_names_in_hg38_v24_kallisto())
    remove_genes = all_genes - tenx_genes

    exp_accs, data_matrix, gene_ids = kqdm.get_gene_tpms_for_experiments(
        exp_accs,
        remove_genes=remove_genes
    )
    data_matrix = np.log(data_matrix+1)
    return exp_accs, data_matrix, gene_ids


def _build_dataset(features, exp_list_f, out_f):
    FEATURE_TO_RETRIEVE = {
        'log_tpm': _retrieve_log_tpm,
        'log_tpm_10x_genes': _retrieve_log_tpm_10x_genes,
        'log_cpm': _retrieve_log_cpm,
        'counts': _retrieve_counts
    }
    with open(exp_list_f, 'r') as f:
        the_exps = json.load(f)['experiments']
    ret_func = FEATURE_TO_RETRIEVE[features]
    ret_exps, data_matrix, gene_ids = ret_func(the_exps) 
    ret_exps = [x.encode('utf-8') for x in ret_exps]   
    gene_ids = [x.encode('utf-8') for x in gene_ids]
    
    # Write data to HDF5
    print('Writing data to {}...'.format(out_f))
    with h5py.File(out_f, 'w') as f:
        f.create_dataset('expression', data=data_matrix, compression="gzip")
        f.create_dataset('experiment', data=np.array(ret_exps))
        f.create_dataset('gene_id', data=np.array(gene_ids))
    print('done.')

if __name__ == "__main__":
    main()
