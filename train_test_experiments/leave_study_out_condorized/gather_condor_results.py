from optparse import OptionParser
import os
import pandas as pd
from os.path import isdir, join

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    condor_root = args[0]
    out_filename = args[1]
    out_f = options.out_file

    # Loading all Condor output files
    dfs = []
    for job_root in os.listdir(condor_root):
        if not isdir(join(condor_root, job_root)):
            continue
        job_out_f = join(condor_root, job_root, out_filename)
        #print("Retrieving data from {}...".format(job_out_f))
        df = pd.read_csv(job_out_f, sep='\t', index_col=0)
        if 'SRX2433569' in df.index:
            print('FOUND IT IN {}'.format(job_out_f))
        dfs.append(df)
    # Determining the columns common to all dataframes
    common_cols = set(dfs[0].columns)
    for df in dfs:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)
    # Aggregating dataframes
    dfs = [
        df[common_cols]
        for df in dfs
    ]
    agg_df = pd.concat(dfs)
    print('Aggregated results:\n{}'.format(agg_df))
    print('Writing to {}.'.format(out_f))
    agg_df.to_csv(out_f, sep='\t')




if __name__ == "__main__":
    main()
