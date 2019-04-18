#############################################################
#   This is the script that is kicked off by the Condor
#   executable
#############################################################
from optparse import OptionParser
import json

import gsea
from gsea import run_gsea

PER_JOB_OUT_F = "gsea_coeffs_results.json"

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    # Read the data in the job's input data file
    input_data_f = args[0]
    gmt_f = args[1]
    with open(input_data_f, 'r') as f:
        input_data = json.load(f)
    gene_names = input_data['gene_names']
    coeffs = input_data['coefficients']
    label = input_data['label']

    # Run GSEA and output the results
    gene_coeff_pairs = zip(gene_names, coeffs)
    tmp_dir = "./gsea_tmp"
    pos_term_to_fdr, neg_term_to_fdr = run_gsea.run_gsea(
        gmt_f, 
        gene_coeff_pairs, 
        tmp_dir
    )
    with open(PER_JOB_OUT_F, 'w') as f:
        f.write(json.dumps(
            {
                'label': label,
                'pos_GO_term_to_FDR': pos_term_to_fdr,
                'neg_GO_term_to_FDR': neg_term_to_fdr
            },
            indent=4
        ))
   

if __name__ == "__main__":
    main()
