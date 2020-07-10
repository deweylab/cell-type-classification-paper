from optparse import OptionParser
import os
from os.path import join
import numpy as np
import scanpy as sc
from anndata import AnnData
import magic
import pandas as pd
import h5py

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="File to write H5 dataset")
    (options, args) = parser.parse_args()

    dataset_f = args[0]
    resolution = float(args[1])
    out_f = options.out_file

    with h5py.File(dataset_f, 'r') as in_f:
        print('Loading expression matrix from {}...'.format(dataset_f))
        X = in_f['expression'][:]
        cell_ids = [
            str(x)[2:-1]
            for x in in_f['experiment'][:]
        ]
        gene_ids = [
            str(x)[2:-1]
            for x in in_f['gene_id'][:]
        ]
        ad = AnnData(
            X=X,
            obs=pd.DataFrame(
                data=cell_ids, 
                columns=['cell']
            ),
            var=gene_ids
        )
        sc.pp.neighbors(ad)
        sc.tl.leiden(ad, resolution=resolution)

        new_X = None
        new_cell_ids = []
        clusters = []
        for clust in sorted(set(ad.obs['leiden'])):
            print('Processing cluster {}'.format(clust))
            indices = [
                int(x) 
                for x in ad.obs.loc[ad.obs['leiden'] == clust].index
            ]
            X = ad.X[indices]               
            print('Shape of cluster matrix: {}'.format(X.shape)) 
            magic_operator = magic.MAGIC()
            magic_X = magic_operator.fit_transform(X)
            if new_X is None:
                new_X = magic_X
            else:
                new_X= np.concatenate([new_X, magic_X])
            print('Current shape of final matrix: {}'.format(new_X.shape))
            clusters += [clust for i in indices]
            new_cell_ids += list(np.array(cell_ids)[indices])

        clusters = [
            x.encode('utf-8')
            for x in clusters
        ]
        new_cell_ids = [
            x.encode('utf-8')
            for x in new_cell_ids
        ]

        print('Writing results to {}...'.format(out_f))
        with h5py.File(out_f, 'w') as out_f:
            out_f.create_dataset(
                'expression', data=new_X, compression="gzip"
            )
            out_f.create_dataset('cluster', data=clusters)
            out_f.create_dataset('experiment', data=new_cell_ids)
            # Copy other datasets to new H5 file
            for k in in_f.keys():
                if k != 'expression' and k != 'experiment':
                    out_f.create_dataset(k,data=in_f[k][:])


if __name__ == "__main__":
    main()
