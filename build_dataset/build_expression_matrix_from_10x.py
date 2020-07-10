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
import pandas as pd
import scipy
from scipy.io import mmread

resource_package = __name__

import kallisto_quantified_data_manager_hdf5_py3 as kqdm

DATA_ROOT = '/tier2/deweylab/mnbernstein/10x_data' # TODO make this configurable

DATA_SET_TO_LOCATION = {
    'CD19_B_cells': join(DATA_ROOT, 'CD19_B_cells/filtered_matrices_mex/hg19'),
    'CD14_monocytes': join(DATA_ROOT, 'CD14_monocytes/filtered_matrices_mex/hg19'),
    'CD34_cells':  join(DATA_ROOT, 'CD34_cells/filtered_matrices_mex/hg19'),
    'CD4_CD25_regulatory_T_cells':  join(DATA_ROOT, 'CD4_CD25_regulatory_T_cells/filtered_matrices_mex/hg19'),
    'CD4_CD45RA_naive_T_cells':  join(DATA_ROOT, 'CD4_CD45RA_naive_T_cells/filtered_matrices_mex/hg19'),
    'CD4_CD45RO_memory_T_cells':  join(DATA_ROOT, 'CD4_CD45RO_memory_T_cells/filtered_matrices_mex/hg19'),
    'CD4_helper_T_cells':  join(DATA_ROOT, 'CD4_helper_T_cells/filtered_matrices_mex/hg19'),
    'CD56_NK_cells':  join(DATA_ROOT, 'CD56_NK_cells/filtered_matrices_mex/hg19'),
    'CD8_CD45RA_naive_T_cells':  join(DATA_ROOT, 'CD8_CD45RA_naive_T_cells/filtered_matrices_mex/hg19'),
    'CD8_T_cells': join(DATA_ROOT, 'CD8_T_cells/filtered_matrices_mex/hg19')
}

DATA_SET_TO_PREFIX = {
    'CD19_B_cells': 'b_cell',
    'CD14_monocytes': 'monocyte',
    'CD34_cells':  'cd34_cell',
    'CD4_CD25_regulatory_T_cells': 'cd4_cd25_regulatory_t_cell',
    'CD4_CD45RA_naive_T_cells': 'cd4_cd45ra_naive_t_cell',
    'CD4_CD45RO_memory_T_cells': 'cd4_cd45ro_memory_t_cell',
    'CD4_helper_T_cells': 'cd4_helper_t_cell',
    'CD56_NK_cells': 'cd56_nk_cell',
    'CD8_CD45RA_naive_T_cells': 'cd8_cd45ra_naive_t_cell',
    'CD8_T_cells': 'cd8_t_cell'

}


def main():
    usage = ""
    parser = OptionParser()
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    num_per_type = int(args[0])
    out_f = options.out_file 

    tenx_genes_f = pr.resource_filename(resource_package, "10x_genes.json")
    with open(tenx_genes_f, 'r') as f:
        targ_genes = set(json.load(f))
    print('{} total target genes'.format(len(targ_genes)))
    datasets, barcodes, cell_ids, data_matrix = _load_random_subset(num_per_type, targ_genes)

    gene_ids = [
        gene.encode('utf-8')
        for gene in targ_genes
    ]
    datasets = [
        dataset.encode('utf-8')
        for dataset in datasets
    ]
    barcodes = [
        barcode.encode('utf-8')
        for barcode in barcodes
    ]
    cell_ids = [
        cell_id.encode('utf-8')
        for cell_id in cell_ids
    ]

    print('Writing data to {}...'.format(out_f))
    with h5py.File(out_f, 'w') as f:
        f.create_dataset('expression', data=data_matrix, compression="gzip")
        f.create_dataset('experiment', data=np.array(cell_ids))
        f.create_dataset('barcode', data=np.array(barcodes))
        f.create_dataset('dataset', data=np.array(datasets))
        f.create_dataset('gene_id', data=np.array(gene_ids))
    

def _load_random_subset(num_per_type, targ_genes):
    np.random.seed(8)
    all_barcodes = []
    all_cell_ids = []
    all_datasets = []
    label_sets = [] # TODO
    data_matrix = None
    for cell_type, root in DATA_SET_TO_LOCATION.items():

        # Load genes and map each gene to its index
        fname = join(root, 'genes.tsv')
        tenx_genes = []
        with open(fname, 'r') as f:
            tenx_genes = [
                l.split()[0].strip()
                for l in f
            ]
        tenx_gene_to_index = {
            gene: i
            for i, gene in enumerate(tenx_genes)
        }
        tenx_gene_indices = [
            tenx_gene_to_index[gene]
            for gene in targ_genes
        ]

        # Randomly sample barcodes
        fname = join(root, 'barcodes.tsv')
        barcodes_df = pd.read_csv(fname, header=None)
        n_cells = len(barcodes_df)
        all_indices = np.arange(n_cells)
        rand_indices = np.random.choice(all_indices, num_per_type, replace=False)
        rand_indices = sorted(rand_indices)
        rand_barcodes = list(barcodes_df.iloc[rand_indices][0])
        assert len(set(rand_barcodes)) == num_per_type

        # Keep track of cell barcodes and dataset that each cell 
        # came from
        all_barcodes += rand_barcodes
        all_datasets += list(np.full(
            len(rand_barcodes),
            cell_type
        ))
    
        # Create interpretable cell_ids
        for i in range(len(rand_barcodes)):
            all_cell_ids.append('%s.%d' % (DATA_SET_TO_PREFIX[cell_type], i))
        
        # Load the counts matrix and concatenate to current
        # data matrix
        fname = join(root, 'matrix.mtx')
        print('Loading counts from {}...'.format(fname))
        with open(fname, 'rb') as f:
            mat = mmread(f).todense().T
        mat = mat[rand_indices,:]
        mat = mat[:,tenx_gene_indices]
        if data_matrix is None:
            data_matrix = mat
        else:
            data_matrix = np.concatenate([data_matrix, mat])
        
    # Compute log-cpm
    data_matrix = np.array(data_matrix, dtype=np.float64)
    #print(data_matrix.shape)
    #print('Computing log-CPM')
    #data_matrix = np.array([
    #    x/sum(x)
    #    for x in data_matrix
    #])
    #data_matrix *= 1e6
    #data_matrix = np.log(data_matrix + 1)
    #print('done.')
    #print(data_matrix)

    print('Loaded counts matrix of shape {}'.format(data_matrix.shape)) 
    return all_datasets, all_barcodes, all_cell_ids, data_matrix


if __name__ == "__main__":
    main()
