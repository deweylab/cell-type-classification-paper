#########################################################################
#   Runs Kallisto in order to generate the expression-profile that is
#   used as a feature-vector in the cell type classifier.
#########################################################################

from collections import defaultdict
import os
from os.path import dirname, join, realpath
import subprocess
from optparse import OptionParser
import pkg_resources as pr
import math
import json
import numpy as np

from machine_learning import learners

KALLISTO_BIN = "kallisto"
FRAGMENT_LENGTH = 200
FRAGMENT_STDEV = 30

resource_package = __name__
GENE_TRANSCRIPT_MAPPING_F = pr.resource_filename(
    resource_package, join("resources", "gene_transcript.hg38_p10.tsv")) 
GENE_ORDER = pr.resource_filename(
    resource_package, join("resources", "gene_order.json"))
KALLISTO_REF = pr.resource_filename(
    resource_package, join("resources", "kallisto_reference.hg38_v27"))

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    parser.add_option("-p", "--is_paired", 
        action="store_true", help="Paired-end reads")
    parser.add_option("-o", "--out", help="Output file")
    (options, args) = parser.parse_args()

    fastq_fs = args[0].split(',')
    tmp = args[1]
    is_paired = options.is_paired
    out_f = options.out

     # map each gene to its transcripts
    gene_to_trans = _load_transcript_to_gene_map(
        GENE_TRANSCRIPT_MAPPING_F)
    tran_to_gene = {}
    for gene, trans in gene_to_trans.iteritems():
        for tran in trans:
            tran_to_gene[tran] = gene

    with open(GENE_ORDER, 'r') as f:
        gene_order = json.load(f)

    _run_kallisto(KALLISTO_REF, fastq_fs, is_paired, tmp)

    # Obtain gene-level counts
    kallisto_target_to_count = parse_kallisto_out(
        join(tmp, 'abundance.tsv')
    )
    gene_to_count = _aggregate_by_gene(
        kallisto_target_to_count, tran_to_gene, gene_order
    )
   
    # Create log-CPM vector 
    count_vec = np.array([
        gene_to_count[gene]
        for gene in gene_order
    ])
    log_cpm_vec = count_vec / sum(count_vec)
    log_cpm_vec *= 1000000
    log_cpm_vec = np.log(log_cpm_vec + 1)
    
    with open(out_f, 'w') as f:
        f.write('\t'.join([str(x) for x in log_cpm_vec]))


def _run_kallisto(reference_f, fastq_fs, is_paired, 
    out_dir, stranded=False):
    cmd = "%s quant --index %s --output-dir %s " % (
        KALLISTO_BIN,
        reference_f,
        out_dir
    )
    if is_paired:
        if stranded:
            cmd += " --fr-stranded " 
    else:
        cmd += "--single -l %d -s %d " % (
            FRAGMENT_LENGTH,
            FRAGMENT_STDEV
        )
    cmd += " ".join(fastq_fs)
    subprocess.call(cmd, shell=True, env=None)


def parse_kallisto_out(kallisto_out_f):
    """
    The first few lines of a Kallisto output file look like:
        target_id       length  eff_length      est_counts      tpm
        ENST00000373020.8       2206    2007    0       0
        ENST00000494424.1       820     621     0       0
        ENST00000496771.5       1025    826     0       0
        ENST00000612152.4       3796    3597    64.4682 5.17246
        ENST00000614008.4       900     701     0       0
        ENST00000373031.4       1339    1140    1       0.253155
        ENST00000485971.1       542     343     1       0.841391
    """
    target_to_count = {}
    with open(kallisto_out_f, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            toks = line.split()
            target = toks[0]
            count = float(toks[3])
            tpm = float(toks[4])
            if math.isnan(count):
                print "Warning! Found a NaN count in file %s" % kallisto_out_f
                return None
            target_to_count[target] = count
    return target_to_count   
 

def _aggregate_by_gene(kallisto_target_to_count, tran_to_gene, gene_order):
    """
    Given a data matrix where the rows are samples/experiments
    and the columns are transcripts, compute a new data matrix
    where for each row and for each gene, we sum the values in
    that row for that gene.
    
    Args:
        data_matrix: an [N,T] matrix where N is the number of
            experiments and T is the number of transcripts
    Returns:
        gene_data_matrix: an [N,G] matrix where N is the number
            of experiments and G is the number of genes
        gene_names: the gene identifier corresponding to each 
            column of the gene_data_matrix
    """
    # map genes to the kallisto targets for that gene
    kallisto_gene_to_targets = _compute_kallisto_gene_to_targets(
        kallisto_target_to_count.keys(),
        tran_to_gene
    )

    gene_to_count = {}
    for gene in gene_order:
        targets = kallisto_gene_to_targets[gene]
        count = sum([
            kallisto_target_to_count[target]
            for target in targets
        ])
        gene_to_count[gene] = count
    return gene_to_count


def _compute_kallisto_gene_to_targets(kallisto_targets, tran_to_gene):
    """
    Each target in the Kallisto output has a prefix,
    which is the transcript ID, and a suffix, which is the
    version number. Example:
        ENST00000624481.4_PAR_Y
        ENST00000624481.4
    We first map each transcript ID (e.g. 'ENST00000624481') 
    to its set of target ID's (e.g. {'ENST00000624481.4_PAR_Y', 
     'ENST00000624481.4'}). Then, we map each gene to this set
    of targets so in order to form a mapping from genes to
    entries in the Kallisto expression profile.
    
    Args:
        kallisto_targets: the list of names of each genomic
            target output by Kallisto. These are transcripts
            and are formed by taking a cannonical transcript
            name and appending some data to it (e.g. the 
            cannonical transcript name 'ENST00000624481' may
            have an entry called 'ENST00000624481.4_PAR_Y'  
            in the Kallisto output. 
        tran_to_gene: a dictionary mapping each cannonical
            transcript name to its gene
    """
    kallisto_tran_to_targets = defaultdict(lambda: set())
    for target in kallisto_targets:
        tran = target.split(".")[0]
        kallisto_tran_to_targets[tran].add(target)
    kallisto_tran_to_targets = dict(kallisto_tran_to_targets)

    kallisto_gene_to_targets = defaultdict(lambda: set())
    for tran, targets in kallisto_tran_to_targets.iteritems():
        gene = tran_to_gene[tran]
        kallisto_gene_to_targets[gene].update(targets)
    kallisto_gene_to_targets = dict(kallisto_gene_to_targets)
    return kallisto_gene_to_targets

def _load_transcript_to_gene_map(gene_transcript_tsv_f):
    gene_to_transs = defaultdict(lambda: set())
    with open(gene_transcript_tsv_f, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            toks = line.split()
            gene = toks[0]
            trans = toks[1]
            gene_to_transs[gene].add(trans)
    return gene_to_transs


if __name__ == "__main__":
    main()


