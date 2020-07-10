from optparse import OptionParser
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
from os.path import join
import pandas as pd

from common import load_dataset

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Output directory")
    (options, args) = parser.parse_args()
    
    dataset = args[0]
    out_dir = options.out_dir
    
    r = load_dataset.load_dataset(dataset, 'counts')
    the_exps = r[3]
    data_matrix = r[10]
    gene_ids = r[11]

    genes_df = pd.read_csv('genes.tsv', sep='\t', index_col=0, header=None)
    genes_df = genes_df.loc[gene_ids]

    with open(join(out_dir, 'matrix.mtx'), 'wb') as f:
        mmwrite(f, coo_matrix(data_matrix.T))
    with open(join(out_dir, 'barcodes.tsv'), 'w') as f:
        f.write('\n'.join(the_exps))    
    genes_df.to_csv(join(out_dir, 'genes.tsv'), sep='\t', header=False)
    
if __name__ == "__main__":
    main()
