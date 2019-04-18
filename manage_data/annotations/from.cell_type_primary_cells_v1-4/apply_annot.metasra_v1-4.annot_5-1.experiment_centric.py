###############################################################################
# Apply a patch to create a phenotyping environment. This script uses a patch
# of the form developed in May 2018 (annotate each study with partitions of 
# samples, where each partition is annotated with a set of tags)
###############################################################################

import os
from os.path import join
import sys

from optparse import OptionParser
import json
import subprocess
import sqlite3
from collections import defaultdict

sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import the_ontology
from map_sra_to_ontology import ontology_graph

ENVIRONMENTS_ROOT = "/tier2/deweylab/mnbernstein/phenotyping_environments"
ONTO_GRAPH_FIG_SCRIPT = "/ua/mnbernstein/projects/tbcp/phenotyping/generate_phenotyping_env/generate_onto_graph_figs.py"
METASRA_DB = "/tier2/deweylab/mnbernstein/metasra/metasra.v1-4.sqlite"
SRA_SUB_DB = "/tier2/deweylab/mnbernstein/sra_metadb/SRAmetadb.subdb.18-02-02.sqlite"
DATA_SET_INFO_FNAME = "data_set_metadata.json"


def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    #patch_f = args[0]
    patch_f = "/ua/mnbernstein/projects/tbcp/phenotyping/manage_data/annotations/from.cell_type_primary_cells_v1-4/cell_type_primary_cells_v1-4.annot_5-1.experiment_centric.json"
    with open(patch_f, 'r') as f:
        patch = json.load(f)

    tags_f = "/ua/mnbernstein/projects/tbcp/phenotyping/manage_data/tags/tags.json"
    new_env_name = "cell_type.metasra_1-4.annot_5-1"
    new_env_root = join(ENVIRONMENTS_ROOT, new_env_name)
    _run_cmd("mkdir %s" % new_env_root)

    # Gather all experiments
    all_experiments = set()
    for annot_data in patch['annotated_studies']:
        for partition in annot_data['partitions']:
            all_experiments.update(partition['experiments'])    

    exp_to_sample = map_experiment_to_sample(all_experiments)

    all_samples = set()
    for exp in all_experiments:
        all_samples.add(exp_to_sample[exp])

    # Map sample to cell type terms
    sample_to_terms = map_samples_to_terms(all_samples)
    sample_to_terms = defaultdict(
        lambda: set(),
        {
            sample: set([
                x
                for x in terms  
                if x[0:2] == "CL"
            ])
            for sample, terms in sample_to_terms.iteritems()
        }
    )

    exp_to_info = {}
    for annot_data in patch['annotated_studies']:
        # Add extra annotated terms to samples
        if 'sample_to_add_terms' in annot_data:
            for sample, add_terms in annot_data['sample_to_add_terms'].iteritems():
                sample_to_terms[sample].update(add_terms)
        # Remove some terms from some samples
        if 'sample_to_remove_terms' in annot_data:
            for sample, rem_terms in annot_data['sample_to_remove_terms'].iteritems():
                sample_to_terms[sample] = list(sample_to_terms[sample] - set(rem_terms))
      
        for partition in annot_data['partitions']:
            for exp in partition['experiments']:
                sample = exp_to_sample[exp]
                exp_to_info[exp] = {
                    "mapped_terms": list(sample_to_terms[sample]),
                    "study_accession": annot_data["study_accession"],
                    "sample_accession": sample,
                    "tags": partition['tags']
                }

    # Write data set info output to new environment
    _run_cmd("mkdir %s" % join(new_env_root, "data"))
    new_data_set_info_f = join(new_env_root, "data", DATA_SET_INFO_FNAME)
    with open(new_data_set_info_f, 'w') as f:
        f.write(json.dumps(
            exp_to_info,
            indent=4,
            separators = (',', ': ')
        ))

    # Create a list of experiments corresponding to all of the
    # experiments
    all_exps_list_dir = join(
        new_env_root, "data", "experiment_lists", "all_experiments"
    )
    _run_cmd("mkdir -p %s" % all_exps_list_dir)
    all_exps_list = {
        "list_name": "all_experiments",
        "description": "These are all of the experiments in this environment's dataset",
        "experiments": list(sorted(exp_to_info.keys()))
    }
    with open(join(all_exps_list_dir, "all_experiments.json"), 'w') as f:
        f.write(json.dumps(
            all_exps_list,
            indent=4
        ))


def map_experiment_to_sample(
        restrict_to_experiments
    ):
    """ 
    Map map each SRA experiment to its sample.
    """
    experiment_sql = """SELECT experiment_accession,
    sample_accession, study_accession FROM experiment
    """
    print "querying database for experiment to sample mappings..."
    exp_to_sample = {}
    with sqlite3.connect(SRA_SUB_DB) as db_conn:
        c = db_conn.cursor()
        returned = c.execute(experiment_sql)
        for r in returned:
            exp = r[0]
            sample = r[1]
            study = r[2]
            if exp in restrict_to_experiments:
                exp_to_sample[exp] = sample
    print "done."
    return exp_to_sample


def map_samples_to_terms(restrict_to_samples):
    og = the_ontology.the_ontology()
    query_metasra_mapped_terms_sql = "SELECT sample_accession, \
        term_id FROM mapped_ontology_terms;"
    print "querying database for sample to terms mappings..."
    sample_to_mapped_terms = defaultdict(lambda: set())
    with sqlite3.connect(METASRA_DB) as metasra_conn:
        metasra_c = metasra_conn.cursor()    
        results = metasra_c.execute(query_metasra_mapped_terms_sql)
        for r in results:
            sample = r[0]
            term_id = r[1]
            if sample in restrict_to_samples:
                sample_to_mapped_terms[sample].add(term_id)
    
    # Restrict to most specific term
    mod_sample_to_mapped_terms = {}
    for sample, terms in sample_to_mapped_terms.iteritems():
        ms_mapped_terms = ontology_graph.most_specific_terms(
            terms, og, sup_relations=["is_a", "part_of"]
        )
        mod_sample_to_mapped_terms[sample] = set(ms_mapped_terms)
    sample_to_mapped_terms = mod_sample_to_mapped_terms
    print "done."
    return sample_to_mapped_terms


def _run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)
    

if __name__ == "__main__":
    main()
