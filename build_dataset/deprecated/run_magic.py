from optparse import OptionParser
import magic
import h5py
import numpy as np

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    parser.add_option("-o", "--out_file", help="File to write output H5 file")
    (options, args) = parser.parse_args()

    dataset_f = args[0]
    out_f = options.out_file

    with h5py.File(dataset_f, 'r') as in_f:
        print('Loading expression matrix from {}...'.format(dataset_f))
        X = in_f['expression'][:]
        print('done.')
        print('Running MAGIC...')
        magic_operator = magic.MAGIC()
        magic_X = magic_operator.fit_transform(X)
        print('done.')
        print('Writing results to {}...'.format(out_f))
        with h5py.File(out_f, 'w') as out_f:
            out_f.create_dataset(
                'expression', data=magic_X, compression="gzip"
            )
            # Copy other datasets to new H5 file
            for k in in_f.keys():
                if k != 'expression':
                    out_f.create_dataset(k,data=in_f[k][:])
            

if __name__ == "__main__":
    main()
