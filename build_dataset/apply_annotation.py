###############################################################################
# Apply a patch to create a phenotyping environment. This script uses a patch
# of the form developed in May 2018 (annotate each study with partitions of 
# samples, where each partition is annotated with a set of tags)
###############################################################################
from optparse import OptionParser
import json
import sqlite3
from collections import defaultdict

from common import the_ontology
from onto_lib_py3 import ontology_graph

EXTRA_SAMPLES = [
    'SRP125125_exp_to_sample.json'
]

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    config_f = args[0]
    out_f = args[1]

    with open(config_f, 'r') as f:
        config = json.load(f)
    annot_f = config['annotation_file']
    tags_f = config['tags_file']
    metasra_f = config['metasra']
    sradb_f = config['sradb']

    with open(annot_f, 'r') as f:
        annot = json.load(f)

    # Gather all experiments
    all_experiments = set()
    for annot_data in annot['annotated_studies']:
        for partition in annot_data['partitions']:
            all_experiments.update(partition['experiments'])    

    exp_to_sample = map_experiment_to_sample(all_experiments, sradb_f)

    all_samples = set()
    for exp in all_experiments:
        all_samples.add(exp_to_sample[exp])

    # Map sample to cell type terms
    sample_to_terms = map_samples_to_terms(all_samples, metasra_f)
    sample_to_terms = defaultdict(
        lambda: set(),
        {
            sample: set([
                x
                for x in terms  
                if x[0:2] == "CL"
            ])
            for sample, terms in sample_to_terms.items()
        }
    )
    sample_to_supp_terms = defaultdict(
        lambda: set(),
        {
            k: set(v)
            for k,v in sample_to_terms.items()
        }
    )

    exp_to_info = {}
    for annot_data in annot['annotated_studies']:
        # Add extra annotated terms to samples
        if 'sample_to_add_terms' in annot_data:
            for sample, add_terms in annot_data['sample_to_add_terms'].items():
                sample_to_terms[sample].update(add_terms)
                sample_to_supp_terms[sample].update(add_terms)
        # Remove some terms from some samples
        if 'sample_to_remove_terms' in annot_data:
            for sample, rem_terms in annot_data['sample_to_remove_terms'].items():
                sample_to_terms[sample] = list(sample_to_terms[sample] - set(rem_terms))
                sample_to_supp_terms[sample] = sample_to_supp_terms[sample] - set(rem_terms)
        if 'sample_to_supplemental_terms' in annot_data:
            for sample, supp_terms in annot_data['sample_to_supplemental_terms'].items():
                sample_to_supp_terms[sample].update(set(supp_terms)) 

        for partition in annot_data['partitions']:
            for exp in partition['experiments']:
                sample = exp_to_sample[exp]
                exp_to_info[exp] = {
                    "mapped_terms": list(sample_to_terms[sample]),
                    "supplemental_mapped_terms": list(sample_to_supp_terms[sample]),
                    "study_accession": annot_data["study_accession"],
                    "sample_accession": sample,
                    "tags": partition['tags']
                }

    # Write data set info output to new environment
    with open(out_f, 'w') as f:
        json.dump(
            exp_to_info,
            f,
            indent=4,
            separators = (',', ': ')
        )


def map_experiment_to_sample(
        restrict_to_experiments,
        sradb_f
    ):
    """ 
    Map map each SRA experiment to its sample.
    """
    experiment_sql = """SELECT experiment_accession,
    sample_accession, study_accession FROM experiment
    """
    print("Querying database for experiment to sample mappings...")
    exp_to_sample = {}
    with sqlite3.connect(sradb_f) as db_conn:
        c = db_conn.cursor()
        returned = c.execute(experiment_sql)
        for r in returned:
            exp = r[0]
            sample = r[1]
            study = r[2]
            if exp in restrict_to_experiments:
                exp_to_sample[exp] = sample
    print("done.")

    # Add mappings for extra experiments/samples that 
    # were not captured in the SRAdb
    for extra_exp_to_sample_f in EXTRA_SAMPLES:
        with open(extra_exp_to_sample_f, 'r') as f:
            extra_exp_to_sample = json.load(f)
        for exp, sample in extra_exp_to_sample.items():
            if exp in restrict_to_experiments:
                exp_to_sample[exp] = sample
    return exp_to_sample 


def map_samples_to_terms(restrict_to_samples, metasra_f):
    og = the_ontology.the_ontology()
    query_metasra_mapped_terms_sql = "SELECT sample_accession, \
        term_id FROM mapped_ontology_terms;"
    print("Querying database for sample to terms mappings...")
    sample_to_mapped_terms = defaultdict(lambda: set())
    with sqlite3.connect(metasra_f) as metasra_conn:
        metasra_c = metasra_conn.cursor()    
        results = metasra_c.execute(query_metasra_mapped_terms_sql)
        for r in results:
            sample = r[0]
            term_id = r[1]
            if sample in restrict_to_samples:
                sample_to_mapped_terms[sample].add(term_id)
    
    # Restrict to most specific term
    mod_sample_to_mapped_terms = {}
    for sample, terms in sample_to_mapped_terms.items():
        ms_mapped_terms = ontology_graph.most_specific_terms(
            terms, og, sup_relations=["is_a", "part_of"]
        )
        mod_sample_to_mapped_terms[sample] = set(ms_mapped_terms)
    sample_to_mapped_terms = mod_sample_to_mapped_terms
    print("done.")
    return sample_to_mapped_terms


    

if __name__ == "__main__":
    main()
