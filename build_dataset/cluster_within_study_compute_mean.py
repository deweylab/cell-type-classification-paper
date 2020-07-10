from optparse import OptionParser
import os
from os.path import join
import numpy as np
import scanpy as sc
from anndata import AnnData
import magic
import pandas as pd
import h5py

from common import load_dataset

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="File to write H5 dataset")
    (options, args) = parser.parse_args()

    dataset_dir = args[0]
    features = args[1]
    resolution = float(args[2])
    out_f = options.out_file
 
    r = load_dataset.load_dataset(dataset_dir, features)
    the_exps = r[3]
    exp_to_index = r[4]
    study_to_exps = r[8]
    X = r[10]
    gene_ids = r[11]
   
    new_X = None
    new_exps = []
    clusters = [] 
    for study_i, (study, exps) in enumerate(study_to_exps.items()):
        print('{}/{} clustering study {} with {} samples...'.format((study_i+1), len(study_to_exps), study, len(exps)))
        exps = list(exps)
        study_indices = [
            exp_to_index[exp]
            for exp in exps
        ]
        X_study = X[study_indices,:]
        if len(exps) > 50:
            ad = AnnData(
                X=X_study,
                obs=pd.DataFrame(
                    data=exps, 
                    columns=['experiment']
                ),
                var=gene_ids
            )
            sc.pp.neighbors(ad)
            sc.tl.leiden(ad, resolution=resolution)
            for clust in sorted(set(ad.obs['leiden'])):
                print('Processing cluster {}'.format(clust))
                clust_indices = [
                    int(x) 
                    for x in ad.obs.loc[ad.obs['leiden'] == clust].index
                ]
                print('{} cells in the cluster.'.format(len(clust_indices)))
                X_clust = X_study[clust_indices,:]
                #X_agg = _agg_TPMs(X_clust)
                X_agg = _mean_TPM(X_clust)
                if new_X is None:
                    new_X = X_agg
                else:
                    new_X= np.concatenate([new_X, X_agg])
                print('Current shape of final matrix: {}'.format(new_X.shape))
                clusters += ['{}_{}'.format(study, clust) for i in clust_indices]
                new_exps += list(np.array(exps)[clust_indices])
        else: 
            #X_agg = _agg_TPMs(X_study)
            X_agg = _mean_TPM(X_study)
            if new_X is None:
                new_X = X_agg
            else:
                new_X= np.concatenate([new_X, X_agg])
            print('Current shape of final matrix: {}'.format(new_X.shape))
            clusters += ['{}_{}'.format(study, '1') for i in range(len(exps))]
            new_exps += exps 

    clusters = [
        x.encode('utf-8')
        for x in clusters
    ]
    new_exps = [
        x.encode('utf-8')
        for x in new_exps
    ]
    gene_ids = [
        gene_id.encode('utf-8')
        for gene_id in gene_ids
    ]

    print('Writing results to {}...'.format(out_f))
    with h5py.File(out_f, 'w') as out_f:
        out_f.create_dataset(
            'expression', data=new_X, compression="gzip"
        )
        out_f.create_dataset('cluster', data=clusters)
        out_f.create_dataset('experiment', data=new_exps)
        out_f.create_dataset('gene_id', data=gene_ids)

def _mean_TPM(X):
    # Convert to TPM
    X = np.exp(X)-1
    x_mean = np.mean(X, axis=0)
    x_mean = np.log(x_mean+1)    
    X_mean = np.full((len(X),len(x_mean)), x_mean)
    return X_mean

def _agg_TPMs(X):
    # Convert to TPM
    X = np.exp(X)-1
    # Sum the TPM's
    x_agg = np.sum(X, axis=0)
    # Re-normalize
    sum_x_agg = float(sum(x_agg))
    x_agg = np.array([x/sum_x_agg for x in x_agg])
    # Compute log(TPM+1)
    x_agg *= 1e6
    x_agg = np.log(x_agg+1)
    # Blow up to full dimensionality
    X_agg = np.full((len(X),len(x_agg)), x_agg)
    return X_agg

if __name__ == "__main__":
    main()
