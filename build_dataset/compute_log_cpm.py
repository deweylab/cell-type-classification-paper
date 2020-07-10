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

        # Normalize
        X = np.array(X, dtype=np.float64)
        print(X.shape)
        print('Computing log-CPM')
        X = np.array([
            x/sum(x)
            for x in X
        ])
        X *= 1e6
        X = np.log(X + 1)
        print('done.')

        with h5py.File(out_f, 'w') as out_f:
            out_f.create_dataset(
                'expression', data=X, compression="gzip"
            )
            # Copy other datasets to new H5 file
            for k in in_f.keys():
                if k != 'expression':
                    out_f.create_dataset(k,data=in_f[k][:])


if __name__ == "__main__":
    main()
