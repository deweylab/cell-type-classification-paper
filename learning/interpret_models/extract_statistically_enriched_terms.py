from optparse import OptionParser
import json
import sys
import collections
from collections import defaultdict

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import the_ontology


FDR_THRESH_1 = 0.05
FDR_THRESH_2 = 0.1

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_f", help="Output file")
    (options, args) = parser.parse_args()

    gsea_results_f = args[0]
    out_f = options.out_f

    with open(gsea_results_f, 'r') as f:
        gsea_results = json.load(f)
    label_to_pos_term_to_fdr = gsea_results['label_to_pos_term_to_fdr']
    label_to_neg_term_to_fdr = gsea_results['label_to_neg_term_to_fdr']

    cl_og = the_ontology.the_ontology()
    go_og = the_ontology.go_ontology()

    label_to_enriched_gos = {}
    for label, term_to_fdr in label_to_pos_term_to_fdr.iteritems():
        label_to_enriched_gos[label] = {
            "FDR < %f" % FDR_THRESH_1: [],
            "FDR < %f" % FDR_THRESH_2: [],
            "Cell type": cl_og.id_to_term[label].name
        }
        for term, fdr in term_to_fdr.iteritems():
            if fdr < FDR_THRESH_1:
                label_to_enriched_gos[label]["FDR < %f" % FDR_THRESH_1].append({
                    "GO term": term,
                    "GO term name": go_og.id_to_term[term].name,
                    "FDR": fdr,
                    "Enrichment direction": "positive"
                })
            elif fdr < FDR_THRESH_2:
                label_to_enriched_gos[label]["FDR < %f" % FDR_THRESH_2].append({
                    "GO term": term,
                    "GO term name": go_og.id_to_term[term].name,
                    "FDR": fdr,
                    "Enrichment direction": "positive"
                })
    for label, term_to_fdr in label_to_neg_term_to_fdr.iteritems():
        for term, fdr in term_to_fdr.iteritems():
            if fdr < FDR_THRESH_1:
                label_to_enriched_gos[label]["FDR < %f" % FDR_THRESH_1].append({
                    "GO term": term,
                    "GO term name": go_og.id_to_term[term].name,
                    "FDR": fdr,
                    "Enrichment direction": "negative"
                })
            elif fdr < FDR_THRESH_2:
                label_to_enriched_gos[label]["FDR < %f" % FDR_THRESH_2].append({
                    "GO term": term,
                    "GO term name": go_og.id_to_term[term].name,
                    "FDR": fdr,
                    "Enrichment direction": "negative"
                })

    with open(out_f, 'w') as f:
        f.write(json.dumps(
            label_to_enriched_gos,
            indent=4
        ))


if __name__ == "__main__":
    main()
