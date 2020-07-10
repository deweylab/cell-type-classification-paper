from optparse import OptionParser
from collections import defaultdict
import sys
import os
from os.path import join
import json
import pandas as pd

from common import the_ontology

SCMATCH_OUTPUT_TO_TERMS = {
    'CD4+ T Cells': [
        'CL:0000624'    # CD4-positive, alpha-beta T cell 
    ],
    'CD4+CD25+CD45RA- memory regulatory T cells expanded': [
        'CL:0002678'    # memory regulatory T cell 
    ],
    'CD133+ stem cells - adult bone marrow derived': [
        'CL:0000037'    # hematopoietic stem cell
    ],
    'CD133+ stem cells - adult bone marrow derived, pool1.CNhs12552.12224-129F1': [
        'CL:0000037'    # hematopoietic stem cell
    ], 
    'CD4+CD25+CD45RA- memory regulatory T cells': [
        'CL:0002678'    # memory regulatory T cell 
    ], 
    'CD4+CD25-CD45RA+ naive conventional T cells expanded': [
        'CL:0000895'    # naive thymus-derived CD4-positive, alpha-beta T cell
    ], 
    'CD4+CD25-CD45RA+ naive conventional T cells': [
        'CL:0000895'    # naive thymus-derived CD4-positive, alpha-beta T cell
    ], 
    'CD4+CD25-CD45RA- memory conventional T cells': [
        'CL:0000897'    # CD4-positive, alpha-beta memory T cell 
    ], 
    'CD4+CD25-CD45RA- memory conventional T cells expanded': [
        'CL:0000897'    # CD4-positive, alpha-beta memory T cell 
    ], 
    'CD34+ stem cells - adult bone marrow derived': [
        'CL:0000037'    # hematopoietic stem cell
    ], 
    'CD14+CD16+ Monocytes': [
        'CL:0002397'    # CD14-positive, CD16-positive monocyte
    ], 
    'CD4+CD25+CD45RA+ naive regulatory T cells': [
        'CL:0000895',   # naive thymus-derived CD4-positive, alpha-beta T cell
        'CL:0002677'    # naive regulatory T cell
    ], 
    'CD8+ T Cells (pluriselect)': [
        'CL:0000625'    # CD8-positive, alpha-beta T cell
    ],
    'Monocyte-derived macrophages response to mock influenza infection': [
        'CL:0000235'    # macrophage
    ], 
    'CD4+CD25+CD45RA+ naive regulatory T cells expanded': [
        'CL:0000895',   # naive thymus-derived CD4-positive, alpha-beta T cell
        'CL:0002677'    # naive regulatory T cell
    ],
    'gamma delta positive T cells': [
        'CL:0000798'    # gamma-delta T cell
    ], 
    'CD8+ T Cells': [
        'CL:0000625'    # CD8-positive, alpha-beta T cell
    ], 
    'Natural Killer Cells': [
        'CL:0000623'    # natural killer cell
    ], 
    'Dendritic Cells - plasmacytoid': [
        'CL:0000784'    # plasmacytoid dendritic cell
    ], 
    'CD14-CD16+ Monocytes': [
        'CL:0002396'    # CD14-low, CD16-positive monocyte
    ],
    'CD19+ B Cells (pluriselect)': [
        'CL:0000236'    # B cell
    ],
    'CD34+ Progenitors': [
        'CL:0008001'    # hematopoietic precursor cell
    ],
    'Basophils': [
        'CL:0000767'    # basophil
    ], 
    'Monocyte-derived macrophages response to udorn influenza infection': [
        'CL:0000235'    # macrophage
    ],
    'CD14+CD16- Monocytes': [
        'CL:0002057'    # CD14-positive, CD16-negative classical monocyte
    ],
    'CD19+ B Cells': [
        'CL:0000236'    # B cell
    ],
    'Aortic smooth muscle cell response to IL1b': [
        'CL:0002539'    # aortic smooth muscle cell
    ],
    'chorionic membrane cells': [
        'CL:0002371'    # somatic cell # TODO is this CL:0002541?
    ],
    'Mesenchymal Stem Cells - umbilical': [
        'CL:0000134'    # mesenchymal stem cell
    ],
    'immature langerhans cells': [
        'CL:0000453'    # Langherans cell
    ],
    'mesenchymal precursor cell - adipose': [
        'CL:0002570'    # mesenchymal stem cell of adipose
    ],
    'Chondrocyte - re diff': [
        'CL:0000138'    # chondrocyte
    ],
    'Renal Proximal Tubular Epithelial Cell': [
        'CL:0002306'    # epithelial cell of proximal tubule
    ],
    'Preadipocyte - breast': [
        'CL:0002334'    # preadipocyte
    ],
    'Smooth Muscle Cells - Pulmonary Artery': [
        'CL:0002591'    # smooth muscle cell of the pulmonary artery
    ],
    'Mesothelial Cells': [
        'CL:0000077'    # mesothelial cell
    ],
    'Fibroblast - Choroid Plexus': [
        'CL:0002549'    # fibroblast of choroid plexus
    ],
    'H9 Embryonic Stem cells': [
        'CL:0002322'    # embryonic stem cell
    ],
    'Renal Epithelial Cells': [
        'CL:0002518'    # kidney epithelial cell
    ],
    'migratory langerhans cells': [
        'CL:0000453'    # Langerhans cell
    ],
    'Mesenchymal Stem Cells - amniotic membrane': [
        'CL:0002537',    # amnion mesenchymal stem cell
        'CL:0000349'     # extraembryonic cell
    ],
    "Mesenchymal Stem Cells - Wharton's Jelly": [
        'CL:0002568'    # mesenchymal stem cell of Wharton's jelly
    ],
    'Renal Cortical Epithelial Cells': [
        'CL:0002584'    # renal cortical epithelial cell
    ],
    'Endothelial Cells - Thoracic': [
        'CL:0000115'    # endothelial cell
    ],
    'mature adipocyte': [
        'CL:0000136'    # fat cell
    ],
    'mesenchymal stem cells (adipose derived)': [
        'CL:0000134'    # mesenchymal stem cell
    ],
    'Preadipocyte - omental': [
        'CL:0002334'    # preadipocyte
    ],
    'Lymphatic Endothelial cells response to VEGFC': [
        'CL:0002138'    # endothelial cell of lymphatic vessel
    ],
    'mesenchymal precursor cell - bone marrow': [
        'CL:0002540'    # mesenchymal stem cell of the bone marrow
    ],
    'Endothelial Cells - Microvascular': [
        'CL:2000008'    # microvascular endothelial cell
    ],
    'Hepatocyte': [
        'CL:0000182'    # hepatocyte
    ],
    'Melanocyte - light': [
        'CL:0002567'    # light melanocyte
    ],
    'Prostate Epithelial Cells': [
        'CL:0002231'    # epithelial cell of prostate
    ],
    'Adipocyte - omental': [
        'CL:0002615'    # adipocyte of omentum tissue
    ],
    'Endothelial Cells - Lymphatic': [
        'CL:0002138'    # endothelial cell of lymphatic vessel
    ],
    'Mast cell - stimulated': [
        'CL:0000097'    # mast cell
    ],
    'Fibroblast - Periodontal Ligament': [
        'CL:2000017'    # fibroblast of peridontal ligament
    ],
    'Lens Epithelial Cells': [
        'CL:0002224'    # lense epithelial cell
    ],
    'Placental Epithelial Cells': [
        'CL:0002577'    # placental epithelial cell
    ],
    'amniotic membrane cells': [
        'CL:0000349'    # extraembryonic cell
    ],
    'Dendritic Cells - monocyte immature derived': [
        'CL:0001056'    # dendritic cell
    ],
    'Smooth Muscle Cells - Brain Vascular': [
        'CL:0002590'    # smooth muscle cell of the brain vasculature
    ],
    'Intestinal epithelial cells (polarized)': [
        'CL:0002563'    # intestinal epithelial cell
    ],
    'Astrocyte - cerebral cortex': [
        'CL:0000127'    # astrocyte
    ],
    'Preadipocyte - perirenal': [
        'CL:0002334'    # preadipocyte
    ],
    'CD14+ monocyte derived endothelial progenitor cells': [
        'CL:0002619'    # adult endothelial progenitor cell
    ],
    'Hepatic Sinusoidal Endothelial Cells': [
        'CL:1000398'    # endothelial cell of hepatic sinusoid
    ],
    'Macrophage - monocyte derived': [
        'CL:0000235'    # macrophage
    ],
    'Retinal Pigment Epithelial Cells': [
        'CL:0002586'    # retinal pigment epithelial cell
    ],
    'Monocyte-derived macrophages response to LPS': [
        'CL:0000235'    # macrophage
    ],
    'Endothelial Cells - Umbilical vein': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],
    'Iris Pigment Epithelial Cells': [
        'CL:0002565'    # iris pigment epithelial cell
    ],
    'Small Airway Epithelial Cells': [
        'CL:0002368'    # respiratory epithelial cell
    ],
    'Peripheral Blood Mononuclear Cells': [
        'CL:2000001'    # peripheral blood mononuclear cell
    ],
    'Fibroblast - Cardiac': [
        'CL:0002548'    # fibroblast of cardiac tissue
    ],
    'Ciliary Epithelial Cells': [
        'CL:0000067'    # ciliated epithelial cell
    ],
    'Aortic smooth muscle cell response to FGF2': [
        'CL:0002539'    # aortic smooth muscle cell
    ],
    'Mast cell': [
        'CL:0000097'    # mast cell
    ],
    'Smooth Muscle Cells - Bronchial': [
        'CL:0002598'    # bronchial smooth muscle cell
    ],
    'Neurons': [
        'CL:0000540'    # neuron
    ],
    'Smooth muscle cells - airway': [   
        'CL:0000192'    # smooth muscle cell
    ],
    'Endothelial Cells - Vein': [
        'CL:0002543'    # vein endothelial cell
    ]
}


def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write output")
    (options, args) = parser.parse_args()

    result_f = args[0]
    scmatch_onto_f = args[1]
    out_dir = options.out_dir

    og = the_ontology.the_ontology()

    scmatch_output_to_all_terms = defaultdict(lambda: set())
    all_terms = set()

    scmatch_output_to_terms = _parse_scmatch_to_output_terms(scmatch_onto_f)
    #all_terms = set()
    #for terms in scmatch_output_to_all_terms.values():
    #    all_terms.update(terms)
    #print(all_terms)
    
    for scmatch_out, terms in scmatch_output_to_terms.items():
        for term in terms:
            scmatch_output_to_all_terms[scmatch_out].update(
                og.recursive_superterms(term)
            )
            all_terms.update(
                og.recursive_superterms(term)
            )
    scmatch_output_to_all_terms = dict(scmatch_output_to_all_terms)
    all_terms = sorted(all_terms)

    print('All terms:')
    print(','.join(all_terms))

    results_df = pd.read_csv(result_f, index_col=0, quotechar='"')
    print(results_df)
    conf_da = []
    bin_da = []
    nonmapped_samples = set()
    for cell in results_df.index:
        scmatch_out = results_df.loc[cell]['top sample'] #.split(',')[0]
        score = results_df.loc[cell]['top correlation score']
        try:
            terms = scmatch_output_to_all_terms[scmatch_out]
        except KeyError:
            nonmapped_samples.add(scmatch_out)
            terms = []
        term_scores = []
        term_assigns = []
        for term in all_terms:
            if term in terms:
                term_scores.append(score)
                term_assigns.append(1)
            else:
                term_scores.append(float('-inf'))
                term_assigns.append(0)
        conf_da.append(term_scores)
        bin_da.append(term_assigns)
    print('Could not the following samples to ontology terms:')
    print('\n'.join(nonmapped_samples)) 
    conf_df = pd.DataFrame(
        data=conf_da,
        columns=all_terms,
        index=results_df.index
    )
    bin_df = pd.DataFrame(
        data=bin_da,
        columns=all_terms,
        index=results_df.index
    )
    conf_df.to_csv(join(out_dir, 'classification_results.tsv'), sep='\t')
    bin_df.to_csv(join(out_dir, 'binary_classification_results.tsv'), sep='\t')
    


def _parse_scmatch_to_output_terms(scmatch_onto_f):
    with open(scmatch_onto_f, 'r') as f:
        onto_data = json.load(f)
    sample_to_term = defaultdict(lambda: [])
    for term_str, samples in onto_data.items():
        term = term_str.split('!')[0].strip()
        if 'CL' in term:
            for sample in samples:
                sample_to_term[sample].append(term)
    print(sample_to_term)
    return sample_to_term    

if __name__ == "__main__":
    main()
