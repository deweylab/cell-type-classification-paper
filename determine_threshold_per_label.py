from optparse import OptionParser
import pandas as pd
import json
import sys

from common import the_ontology

REMOVE_TERMS = set([
    'CL:0000000',
    'CL:0000003',
    'CL:0000010'  
])

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    pr_curves_f = args[0]
    out_f = options.out_file

    og = the_ontology.the_ontology()
    
    with open(pr_curves_f, 'r') as f:
        label_to_pr_curves = json.load(f)
        
    da = []
    for label, pr in label_to_pr_curves.items():
        if label in REMOVE_TERMS:
            continue
        precs = pr[0]
        recs = pr[1]
        threshs = pr[2]
        f1s = map(_compute_f1, zip(precs, recs))
        max_f1_thresh = max(zip(f1s, precs, threshs), key=lambda x: x[0])
        thresh = min([max_f1_thresh[2], 0.5])
        #thresh = max_f1_thresh[2]

        #da.append((label, og.id_to_term[label].name, max_f1_thresh[1], max_f1_thresh[0]))
        da.append((label, og.id_to_term[label].name, thresh, max_f1_thresh[2], max_f1_thresh[1], max_f1_thresh[0]))    

    df = pd.DataFrame(
        data=da, 
        columns=['label', 'label_name', 'threshold', 'empirical_threshold', 'precision', 'F1-score']
    )
    df.to_csv(out_f, sep='\t', index=False)
    print(df) 
                

def _compute_f1(r):
    prec = r[0]
    rec = r[1]
    try:
        f1 = 2 * ((prec * rec)/(prec + rec))    
    except ZeroDivisionError:
        f1 = 0.0
    return f1
    

if __name__ == "__main__":
    main()
