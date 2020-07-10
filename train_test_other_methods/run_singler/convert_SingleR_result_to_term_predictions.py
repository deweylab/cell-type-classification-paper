from optparse import OptionParser
from collections import defaultdict
import sys
import os
from os.path import join
import json
import pandas as pd

from common import the_ontology

SINGLER_OUTPUT_TO_TERMS = {
    'scores.Class.switched.memory.B.cells': [
        'CL:0000972'    # class switched memory B cell
    ],  
    "scores.Neutrophils": [
        "CL:0000775"    # neutrophil
    ], 
    "scores.Monocytes": [
        "CL:0000576"    # monocyte
    ], 
    "scores.HSC": [
        "CL:0000037"    # hematopoietic stem cell
    ], 
    "scores.CD4..T.cells": [
        "CL:0000624"    # CD4-positive, alpha-beta T cell
    ], 
    'scores.CD8..T.cells': [
        "CL:0000625"    # CD8-positive, alpha-beta T cell
    ], 
    'scores.NK.cells': [
        'CL:0000623'    # natural killer cell
    ], 
    'scores.B.cells': [
        'CL:0000236'    # B cell
    ], 
    'scores.Macrophages': [
        'CL:0000235'    # macrophage
    ], 
    'scores.Erythrocytes': [
        'CL:0000232'    # erythrocyte 
    ], 
    'scores.Endothelial.cells': [
        'CL:0000115'    # endothelial cell
    ], 
    'scores.DC': [
        'CL:0000451'    # dendritic cell
    ], 
    'scores.Eosinophils': [
        'CL:0000771'    # eosinophil
    ], 
    'scores.Chondrocytes': [
        "CL:0000138"    # chondrocyte
    ], 
    'scores.Fibroblasts': [
        "CL:0000057"    # fibroblast
    ], 
    'scores.Smooth.muscle': [
        "CL:0000192"    # smooth muscle cell
    ], 
    'scores.Epithelial.cells': [
        "CL:0000066"    # epithelial cell
    ], 
    'scores.Melanocytes': [
        "CL:0000148"    # melanocyte
    ], 
    'scores.Skeletal.muscle': [
        "CL:0000188"    # cell of skeletal muscle
    ], 
    'scores.Keratinocytes': [
        "CL:0000312"    # keratinocyte
    ], 
    'scores.Myocytes': [
        "CL:0000187"    # muscle cell
    ], 
    'scores.Adipocytes': [
        "CL:0000136"    # fat cell
    ], 
    'scores.Neurons': [
        "CL:0000540"    # neuron
    ], 
    'scores.Pericytes': [
        "CL:0000669"    # pericyte cell
    ], 
    'scores.Mesangial.cells': [
        "CL:0000650"    # mesangial cell
    ],
    'scores.Smooth_muscle_cells': [
        "CL:0000192"    # smooth muscle cell
    ], 
    'scores.Epithelial_cells': [
        "CL:0000066"    # epithelial cell
    ], 
    'scores.B_cell': [
        'CL:0000236'    # B cell
    ], 
    'scores.T_cells': [
        'CL:0000084'    # T cell
    ], 
    'scores.Monocyte': [
        "CL:0000576"    # monocyte
    ], 
    'scores.Erythroblast': [
        'CL:0000765'    # erythroblast
    ], 
    'scores.BM...Prog.': [
        'CL:0002092',   # bone marrow cell
        'CL:0008001'    # hematopoietic precursor cell
    ], 
    'scores.Endothelial_cells': [
        'CL:0000115'    # endothelial cell
    ], 
    'scores.Gametocytes': [
        'CL:0000586'    # germ cell
    ], 
    'scores.HSC_.G.CSF': [
        "CL:0000037"    # hematopoietic stem cell
    ], 
    'scores.Macrophage': [
        'CL:0000235'    # macrophage
    ], 
    'scores.NK_cell': [
        'CL:0000623'    # natural killer cell
    ], 
    'scores.Embryonic_stem_cells': [
        'CL:0002322'    # embryonic stem cell
    ], 
    'scores.Tissue_stem_cells': [
        'CL:0000723'    # somatic stem cell
    ], 
    'scores.Osteoblasts': [
        'CL:0000062'    # osteoblast
    ], 
    'scores.Platelets': [
        'CL:0000233'    # platelet
    ], 
    'scores.iPS_cells': [
        'CL:0002248'    # pluripotent stem cell
    ], 
    'scores.Hepatocytes': [
        'CL:0000182'    # hepatocyte
    ], 
    'scores.MSC': [
        'CL:0000134'    # mesenchymal stem cell
    ], 
    'scores.Neuroepithelial_cell': [
        'CL:0000098'    # sensory epithelial cell
    ], 
    'scores.Astrocyte': [
        'CL:0000127'    # astrocyte
    ], 
    'scores.HSC_CD34.': [
        "CL:0000037"    # hematopoietic stem cell
    ], 
    'scores.CMP': [
        'CL:0000049'    # common myeloid progenitor
    ], 
    'scores.GMP': [
        'CL:0000557'    # granulocyte monocyte progenitor cell
    ], 
    'scores.MEP': [
        'CL:0000050'    # megakaryocyte-erythroid progenitor cell
    ], 
    'scores.Myelocyte': [
        'CL:0002193'    # myelocyte
    ], 
    'scores.Pre.B_cell_CD34.': [
        'CL:0000817'    # precursor B cell
    ], 
    'scores.Pro.B_cell_CD34.': [
        'CL:0000826'    # pro-B cell
    ], 
    'scores.Pro.Myelocyte': [
        'CL:0000836'    # promyelocyte
    ],
    'scores.Naive.CD8.T.cells': [
        'CL:0000900'    # naive thymus-derived CD8-positive, alpha-beta T cell
    ],    
    'scores.Central.memory.CD8.T.cells': [
        'CL:0000907'    # central memory CD8-positive, alpha-beta T cell
    ],
    'scores.Effector.memory.CD8.T.cells': [
        'CL:0000913'    # effector memory CD8-positive, alpha-beta T cell
    ],  
    'scores.Terminal.effector.CD8.T.cells': [
        'CL:0001062'    # effector memory CD8-positive, alpha-beta T cell, terminally differentiated
    ],
    'scores.MAIT.cells': [
        'CL:0000940'    # mucosal invariant T cell
    ],
    'scores.Vd2.gd.T.cells': [
        'CL:0000798'    # gamma-delta T cell
    ],
    'scores.Non.Vd2.gd.T.cells': [
        'CL:0000798'    # gamma-delta T cell
    ],
    'scores.Follicular.helper.T.cells': [
        'CL:0002038'    # T follicular helper cell
    ],    
    'scores.T.regulatory.cells': [
        'CL:0000815'    # regulatory T cell
    ],   
    'scores.Th1.cells': [
        'CL:0000545'    # T-helper 1 cell
    ],
    'scores.Th1.Th17.cells': [
        'CL:0000492'    # CD4-positive helper T cell
    ],
    'scores.Th17.cells': [
        'CL:0000899'    # T-helper 17 cell
    ],
    'scores.Th2.cells': [
        'CL:0000546'    # T-helper 2 cell
    ],
    'scores.Naive.CD4.T.cells': [
        'CL:0000895'    # naive thymus-derived CD4-positive, alpha-beta T cell
    ],
    'scores.Progenitor.cells': [
        'CL:0008001'    # hematopoietic precursor cell
    ], 
    'scores.Naive.B.cells': [
        'CL:0000788'    # naive B cell
    ],
    'scores.Non.switched.memory.B.cells': [
        'CL:0000970'    # unswitched memory B cell
    ],  
    'scores.Exhausted.B.cells': [
        'CL:0000236'    # B cell
    ],    
    'scores.Switched.memory.B.cells': [
        'CL:0000972'    # class switched memory B cell
    ],  
    'scores.Plasmablasts': [
        'CL:0000980'    # plasmablast
    ], 
    'scores.Classical.monocytes': [
        'CL:0000860'    # classical monocyte
    ],  
    'scores.Intermediate.monocytes': [
        'CL:0002393'    # intermediate monocyte
    ],
    'scores.Non.classical.monocytes': [
        'CL:0000875'    # non-classical monocyte
    ],
    'scores.Natural.killer.cells': [
        'CL:0000623'    # natural killer cell
    ], 
    'scores.Plasmacytoid.dendritic.cells': [
        'CL:0000784'    # plasmacytoid dendritic cell
    ], 
    'scores.Myeloid.dendritic.cells': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],
    'scores.Low.density.neutrophils': [
        'CL:0000775'    # neutrophil
    ],
    'scores.Low.density.basophils': [
        'CL:0000767'    # basophil
    ],
    'scores.Terminal.effector.CD4.T.cells': [
        'CL:0001044'    # effector CD4-positive, alpha-beta T cell
    ],
    'scores.DC.monocyte.derived.immature': [
        'CL:0001057'    # myeloid dendritic cell, human
    ], 
    'scores.DC.monocyte.derived.Galectin.1': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],   
    'scores.DC.monocyte.derived.LPS': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],
    'scores.DC.monocyte.derived': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],  
    'scores.Smooth_muscle_cells.bronchial.vit_D': [
        'CL:0002598'    # bronchial smooth muscle cell
    ],
    'scores.Smooth_muscle_cells.bronchial': [
        'CL:0002598'    # bronchial smooth muscle cell
    ],
    'scores.Epithelial_cells.bronchial': [
        'CL:0002328'    # bronchial epithelial cell
    ],   
    'scores.B_cell': [
        'CL:0000236'    # B cell
    ],   
    'scores.Neutrophil': [
        'CL:0000775'    # neutrophil
    ],
    'scores.T_cell.CD8._Central_memory': [
        'CL:0000907'    # central memory CD8-positive, alpha-beta T cell
    ],
    'scores.T_cell.CD8.': [
        'CL:0000625'    # CD8-positive, alpha-beta T cell
    ],  
    'scores.T_cell.CD4.': [
        'CL:0000624'    # CD4-positive, alpha-beta T cell
    ],  
    'scores.T_cell.CD8._effector_memory_RA': [
        'CL:0001062'    # effector memory CD8-positive, alpha-beta T cell, terminally differentiated
    ],
    'scores.T_cell.CD8._effector_memory': [
        'CL:0000909'    # CD8-positive, alpha-beta memory T cell 
    ], 
    'scores.T_cell.CD8._naive': [
        'CL:0000900'    # naive thymus-derived CD8-positive, alpha-beta T cell
    ],    
    'scores.Monocyte': [
        'CL:0000576'    # monocyte
    ], 
    'scores.Erythroblast': [
        'CL:0000765'    # erythroblast
    ], 
    'scores.BM': [
        'CL:0002092'    # bone marrow cell
    ],   
    'scores.DC.monocyte.derived.rosiglitazone': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],    
    'scores.DC.monocyte.derived.AM580': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],    
    'scores.DC.monocyte.derived.rosiglitazone.AGN193109': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],  
    'scores.DC.monocyte.derived.anti.DC.SIGN_2h': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],  
    'scores.Endothelial_cells.HUVEC': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],  
    'scores.Endothelial_cells.HUVEC.Borrelia_burgdorferi': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],
    'scores.Endothelial_cells.HUVEC.IFNg': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ], 
    'scores.Endothelial_cells.lymphatic': [
        'CL:0002138'    # endothelial cell of lymphatic vessel
    ],  
    'scores.Endothelial_cells.HUVEC.Serum_Amyloid_A': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],  
    'scores.Endothelial_cells.lymphatic.TNFa_48h': [
        'CL:0002138'    # endothelial cell of lymphatic vessel
    ], 
    'scores.T_cell.effector': [
        'CL:0000911'    # effector T cell
    ],  
    'scores.T_cell.CCR10.CLA.1.25.OH.2_vit_D3.IL.12': [
        'CL:0000084'    # T cell        
    ],  
    'scores.T_cell.CCR10.CLA.1.25.OH.2_vit_D3.IL.12.1': [
        'CL:0000084'    # T cell
    ],    
    'scores.Gametocytes.spermatocyte': [
        'CL:0000017'    # spermatocyte
    ], 
    'scores.DC.monocyte.derived.A._fumigatus_germ_tubes_6h': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],   
    'scores.Neurons.ES_cell.derived_neural_precursor': [
        'CL:0011020'    # neural progenitor cell
    ],
    'scores.Keratinocytes.IL19': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.Keratinocytes.IL20': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.Keratinocytes.IL22': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.Keratinocytes.IL24': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.Keratinocytes.IL26': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.Keratinocytes.KGF': [
        'CL:0000312'    # keratinocyte
    ],    
    'scores.Keratinocytes.IFNg': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.Keratinocytes.IL1b': [
        'CL:0000312'    # keratinocyte
    ],   
    'scores.HSC_.G.CSF': [
        'CL:0008001'    # hematopoietic precursor cell
    ],   
    'scores.DC.monocyte.derived.mature': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],   
    'scores.Monocyte.anti.FcgRIIB': [
        'CL:0000576'    # monocyte
    ],    
    'scores.Macrophage.monocyte.derived.IL.4.cntrl': [
        'CL:0000235'    # macrophage
    ],   
    'scores.Macrophage.monocyte.derived.IL.4.Dex.cntrl': [
        'CL:0000235'    # macrophage
    ],   
    'scores.Macrophage.monocyte.derived.IL.4.Dex.TGFb': [
        'CL:0000235'    # macrophage
    ],    
    'scores.Macrophage.monocyte.derived.IL.4.TGFb': [
        'CL:0000235'    # macrophage
    ],    
    'scores.Monocyte.leukotriene_D4': [
        'CL:0000576'    # monocyte
    ],  
    'scores.NK_cell': [
        'CL:0000623'    # natural killer cell
    ],  
    'scores.NK_cell.IL2': [
        'CL:0000623'    # natural killer cell
    ], 
    'scores.Tissue_stem_cells.iliac_MSC': [
        'CL:0000134'    # mesenchymal stem cell
    ],  
    'scores.Chondrocytes.MSC.derived': [
        'CL:0000138'    # chondrocyte 
    ],
    'scores.Osteoblasts': [
        'CL:0000062'    # osteoblast
    ],  
    'scores.Tissue_stem_cells.BM_MSC': [
        'CL:0000134'    # mesenchymal stem cell 
    ],
    'scores.Osteoblasts.BMP2': [
        'CL:0000062'    # osteoblast 
    ], 
    'scores.Tissue_stem_cells.BM_MSC.BMP2': [
        'CL:0000134'    # mesenchymal stem cell     
    ],    
    'scores.Tissue_stem_cells.BM_MSC.TGFb3': [
        'CL:0000134'    # mesenchymal stem cell 
    ],   
    'scores.DC.monocyte.derived.Poly.IC.': [
        'CL:0001057'    # myeloid dendritic cell, human
    ], 
    'scores.DC.monocyte.derived.CD40L': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],    
    'scores.DC.monocyte.derived.Schuler_treatment': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],    
    'scores.DC.monocyte.derived.antiCD40.VAF347': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],  
    'scores.Tissue_stem_cells.dental_pulp': [
        'CL:0002148',   # dental pulp cell
        'CL:0000723'    # somatic stem cell 
    ],    
    'scores.T_cell.CD4._central_memory': [
        'CL:0000904'    # central memory CD4-positive, alpha-beta T cell 
    ],   
    'scores.T_cell.CD4._effector_memory': [
        'CL:0000905'    # effector memory CD4-positive, alpha-beta T cell
    ],
    'scores.T_cell.CD4._Naive': [
        'CL:0000895'    # naive thymus-derived CD4-positive, alpha-beta T cell
    ],    
    'scores.Smooth_muscle_cells.vascular': [
        'CL:0000359'    # vascular associated smooth muscle cell
    ], 
    'scores.Smooth_muscle_cells.vascular.IL.17': [
        'CL:0000359'    # vascular associated smooth muscle cell
    ],   
    'scores.Platelets': [
        'CL:0000233'    # platelet
    ],    
    'scores.Epithelial_cells.bladder': [
        'CL:0000066'    # epithelial cell 
    ], 
    'scores.Macrophage.monocyte.derived': [
        'CL:0000235'    # macrophage
    ],
    'scores.Macrophage.monocyte.derived.M.CSF': [
        'CL:0000235'    # macrophage
    ],
    'scores.Macrophage.monocyte.derived.M.CSF.IFNg': [
        'CL:0000235'    # macrophage
    ],
    'scores.Macrophage.monocyte.derived.M.CSF.Pam3Cys': [
        'CL:0000235'    # macrophage
    ],    
    'scores.Macrophage.monocyte.derived.M.CSF.IFNg.Pam3Cys': [
        'CL:0000235'    # macrophage
    ],   
    'scores.Macrophage.monocyte.derived.IFNa': [
        'CL:0000235'    # macrophage
    ], 
    'scores.Gametocytes.oocyte': [
        'CL:0000023'    # oocyte
    ],   
    'scores.Monocyte.F._tularensis_novicida': [
        'CL:0000576'    # monocyte
    ],  
    'scores.Endothelial_cells.HUVEC.B._anthracis_LT': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],  
    'scores.B_cell.Germinal_center': [
        'CL:0000844'    # germinal center B cell
    ],
    'scores.B_cell.Plasma_cell': [
        'CL:0000786'    # plasma cell
    ],   
    'scores.B_cell.Naive': [
        'CL:0000788'    # naive B cell
    ], 
    'scores.B_cell.Memory': [
        'CL:0000787'    # memory B cell
    ],
    'scores.DC.monocyte.derived.AEC.conditioned': [
        'CL:0001057'    # myeloid dendritic cell, human
    ],  
    'scores.Tissue_stem_cells.lipoma.derived_MSC': [
        'CL:0000134'    # mesenchymal stem cell
    ],
    'scores.Tissue_stem_cells.adipose.derived_MSC_AM3': [
        'CL:0000134'    # mesenchymal stem cell
    ],    
    'scores.Endothelial_cells.HUVEC.FPV.infected': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],
    'scores.Endothelial_cells.HUVEC.PR8.infected': [    
        'CL:0002618'    # endothelial cell of umbilical vein
    ], 
    'scores.Endothelial_cells.HUVEC.H5N1.infected': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],    
    'scores.Macrophage.monocyte.derived.S._aureus': [
        'CL:0000235'    # macrophage  
    ],    
    'scores.Fibroblasts.foreskin': [
        'CL:1001608'    # foreskin fibroblast 
    ],
    'scores.iPS_cells.skin_fibroblast.derived': [
        'CL:0000034'    # stem cell 
    ],    
    'scores.iPS_cells.skin_fibroblast': [
        'CL:0000034'    # stem cell 
    ],
    'scores.T_cell.gamma.delta ': [
        'CL:0000798'    # gamma-delta T cell 
    ],  
    'scores.Monocyte.CD14.': [
        'CL:0001054'    # CD14-positive monocyte
    ],   
    'scores.Macrophage.Alveolar': [
        'CL:0000583'    # alveolar macrophage
    ],  
    'scores.Macrophage.Alveolar.B._anthacis_spores': [
        'CL:0000583'    # alveolar macrophage
    ],   
    'scores.Neutrophil.inflam': [
        'CL:0000775'    # neutrophil
    ],    
    'scores.iPS_cells.PDB_fibroblasts': [
        'CL:0000034'    # stem cell
    ],    
    'scores.iPS_cells.PDB_1lox.17Puro.5': [
        'CL:0000034'    # stem cell
    ],  
    'scores.iPS_cells.PDB_1lox.17Puro.10': [
        'CL:0000034'    # stem cell
    ], 
    'scores.iPS_cells.PDB_1lox.21Puro.20': [
        'CL:0000034'    # stem cell
    ], 
    'scores.iPS_cells.PDB_1lox.21Puro.26': [
        'CL:0000034'    # stem cell
    ], 
    'scores.iPS_cells.PDB_2lox.5': [
        'CL:0000034'    # stem cell
    ], 
    'scores.iPS_cells.PDB_2lox.22': [
        'CL:0000034'    # stem cell
    ],    
    'scores.iPS_cells.PDB_2lox.21': [
        'CL:0000034'    # stem cell
    ],    
    'scores.iPS_cells.PDB_2lox.17': [
        'CL:0000034'    # stem cell
    ],    
    'scores.iPS_cells.CRL2097_foreskin': [
        'CL:0000034'    # stem cell
    ],   
    'scores.iPS_cells.CRL2097_foreskin.derived.d20_hepatic_diff': [
        'CL:0000034'    # stem cell
    ],  
    'scores.iPS_cells.CRL2097_foreskin.derived.undiff.': [
        'CL:0000034'    # stem cell
    ],   
    'scores.B_cell.CXCR4._centroblast': [
        'CL:0000965'    # Bm3 B cell
    ],    
    'scores.B_cell.CXCR4._centrocyte': [
        'CL:0000966'    # Bm4 B cell
    ], 
    'scores.Endothelial_cells.HUVEC.VEGF': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ], 
    'scores.iPS_cells.fibroblasts': [
        'CL:0000034'    # stem cell
    ],    
    'scores.iPS_cells.fibroblast.derived.Direct_del._reprog': [
        'CL:0000034'    # stem cell
    ],  
    'scores.iPS_cells.fibroblast.derived.Retroviral_transf': [
        'CL:0000034'    # stem cell
    ],   
    'scores.Endothelial_cells.lymphatic.KSHV': [
        'CL:0002138'    # endothelial cell of lymphatic vessel
    ], 
    'scores.Endothelial_cells.blood_vessel': [
        'CL:0000071'    # blood vessel endothelial cell
    ],
    'scores.Monocyte.CD16.': [
        'CL:0000576'    # monocyte
    ],
    'scores.Monocyte.CD16..1': [
        'CL:0000576'    # monocyte
    ], 
    'scores.Tissue_stem_cells.BM_MSC.osteogenic': [
        'CL:0000134'    # mesenchymal stem cell
    ],  
    'scores.Hepatocytes': [
        'CL:0000182'    # hepatocyte
    ],  
    'scores.Neutrophil.uropathogenic_E._coli_UTI89': [
        'CL:0000775'    # neutrophil
    ],   
    'scores.Neutrophil.commensal_E._coli_MG1655': [
        'CL:0000775'    # neutrophil
    ],  
    'scores.MSC': [
        'CL:0000134'    # mesenchymal stem cell
    ],  
    'scores.Neuroepithelial_cell.ESC.derived': [
        'CL:0000098'    # sensory epithelial cell
    ], 
    'scores.Astrocyte.Embryonic_stem_cell.derived': [
        'CL:0000127'    # astrocyte
    ],    
    'scores.Endothelial_cells.HUVEC.IL.1b': [
        'CL:0002618'    # endothelial cell of umbilical vein
    ],
    'scores.HSC_CD34.': [
        'CL:0000037'    # hematopoietic stem cell
    ],
    'scores.B_cell.immature': [
        'CL:0000816'    # immature B cell
    ],  
    'scores.Smooth_muscle_cells.umbilical_vein': [
        'CL:0002588'    # smooth muscle cell of the umbilical vein
    ],   
    'scores.iPS_cells.foreskin_fibrobasts': [
        'CL:0000034'    # stem cell
    ],    
    'scores.iPS_cells.iPS.minicircle.derived': [
        'CL:0000034'    # stem cell
    ], 
    'scores.iPS_cells.adipose_stem_cells': [
        'CL:0000034'    # stem cell
    ], 
    'scores.iPS_cells.adipose_stem_cell.derived.lentiviral': [
        'CL:0000034'    # stem cell
    ],   
    'scores.iPS_cells.adipose_stem_cell.derived.minicircle.derived': [
        'CL:0000034'    # stem cell
    ],   
    'scores.Fibroblasts.breast': [
        'CL:0000057'    # fibroblast
    ],   
    'scores.Monocyte.MCSF': [
        'CL:0000576'    # monocyte
    ],    
    'scores.Monocyte.CXCL4': [
        'CL:0000576'    # monocyte
    ],   
    'scores.Neurons.adrenal_medulla_cell_line': [
        'CL:0000540'    # neuron
    ],    
    'scores.Tissue_stem_cells.CD326.CD56.': [
        'CL:0000723'    # somatic stem cell
    ],    
    'scores.NK_cell.CD56hiCD62L.': [
        'CL:0000623'    # natural killer cell
    ], 
    'scores.T_cell.Treg.Naive': [
        'CL:0002677'    # naive regulatory T cell
    ],    
    'scores.Neutrophil.LPS': [
        'CL:0000775'    # neutrophil
    ],   
    'scores.Neutrophil.GM.CSF_IFNg': [
        'CL:0000775'    # neutrophil
    ],   
    'scores.Monocyte.S._typhimurium_flagellin': [
        'CL:0000576'    # monocyte
    ],    
    'scores.Neurons.Schwann_cell': [
        'CL:0002573'    # Schwann cell
    ],
    'scores.T_cell.gamma.delta': [
        'CL:0000798'    # gamma-delta T cell
    ],
    'scores.Memory.B.cells': [
        'CL:0000787'    # memory B cell
    ],
    'scores.naive.B.cells': [
        'CL:0000788'    # naive B cell
    ],
    'scores.Tregs': [
        'CL:0000815'    # regulatory T cell
    ],    
    'scores.CD4..Tcm': [
        'CL:0000904'    # central memory CD4-positive, alpha-beta T cell
    ],
    'scores.CD4..Tem': [
        'CL:0000905'    # effector memory CD4-positive, alpha-beta T cell
    ],  
    'scores.CD8..Tcm': [
        'CL:0000907'    # central memory CD8-positive, alpha-beta T cell
    ], 
    'scores.CD8..Tem': [
        'CL:0000913'    # effector memory CD8-positive, alpha-beta T cell
    ], 
    'scores.MPP': [
        'CL:0000837'    # hematopoietic multipotent progenitor cell
    ],  
    'scores.CLP': [
        'CL:0000051'    # common lymphoid progenitor
    ],  
    'scores.Megakaryocytes': [
        'CL:0000556'    # megakaryocyte
    ],   
    'scores.Macrophages.M1': [
        'CL:0000863'    # inflammatory macrophage
    ],   
    'scores.Macrophages.M2': [
        'CL:0000890'    # alternatively activated macrophage
    ],
    'scores.Plasma.cells': [
        'CL:0000786'    # plasma cell
    ], 
    'scores.mv.Endothelial.cells': [
        'CL:2000008'    # microvascular endothelial cell
    ],
    'scores.Preadipocytes': [
        'CL:0002334'    # preadipocyte
    ],    
    'scores.Astrocytes': [
        'CL:0000127'    # astrocyte
    ] 
} 


def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write output")
    (options, args) = parser.parse_args()

    result_f = args[0]
    out_dir = options.out_dir

    og = the_ontology.the_ontology()

    raw_df = pd.read_csv(result_f, sep='\t', index_col=0)
    raw_df = raw_df.drop([
        'first.labels', 
        'tuning.scores.first', 
        'tuning.scores.second', 
        'labels', 
        'pruned.labels'
    ], axis=1)

    # Get all terms represented in this output
    all_terms = set()
    for label in raw_df.columns:
        if label not in SINGLER_OUTPUT_TO_TERMS:
            print('Skipping column "{}"'.format(label))
            continue
        all_terms.update(SINGLER_OUTPUT_TO_TERMS[label])

    # Map each term to its ancestors
    term_to_ancestors = {
        term: og.recursive_superterms(term)
        for term in all_terms
    }
    for term, ancestors in term_to_ancestors.items():
        all_terms.update(ancestors)
    all_terms = sorted(all_terms)

    # Compute binary-classification matrix
    da = []
    for cell in raw_df.index:
        preds = [
            (pred, label)
            for pred, label in zip(raw_df.loc[cell], raw_df.columns)
        ]
        max_label = max(preds, key=lambda x: x[0])[1]
        pred_terms = SINGLER_OUTPUT_TO_TERMS[max_label]
        all_pred_terms = set()
        for term in pred_terms:
             all_pred_terms.update(term_to_ancestors[term])

        row = []
        for term in all_terms:
            if term in all_pred_terms:
                row.append(1)
            else:
                row.append(0)
        da.append(row) 
    
    bin_df = pd.DataFrame(
        data=da,
        columns=all_terms,
        index=raw_df.index
    )
    bin_df.to_csv(join(out_dir, 'binary_classification_results.tsv'), sep='\t')
    



            

if __name__ == "__main__":
    main()
