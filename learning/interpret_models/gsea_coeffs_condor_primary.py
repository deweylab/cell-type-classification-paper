#########################################################################
#   Run leave-study-out cross validation. The folds are batched and
#   used as jobs for Condor
#########################################################################

import sys
import os
from os.path import join, basename, isdir
import math
import cPickle
from optparse import OptionParser
import json
import subprocess
import time

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/learning")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/condor_tools")

import condor_submit_tools
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from machine_learning import learners
import leave_study_out as lso
import query_condor_jobs
import project_data_on_ontology as pdoo

THE_CONDOR_TARBALL = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/interpret_models/gsea_coeffs_condor_tarball.tar.gz"
SUBMIT_FNAME = "condor.submit" 
EXECUTABLE_FNAME = "gsea_coeffs_condor_executable.bash"
SUBMIT_LOC = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/interpret_models/gsea_coeffs_condor_executable.bash"
PER_JOB_OUT_F = "gsea_coeffs_results.json"

def main():
    usage = "usage: %prog <config file> <environment dir> <experiment list name>"
    parser = OptionParser(usage)
    parser.add_option("-o", "--out_file", help="File to write experiment results")
    parser.add_option("-c", "--condor_root", help="The directory in which to write temporary files")
    (options, args) = parser.parse_args()

    env_dir = args[0]
    exp_list_name = args[1]
    model_f = args[2]
    #std_coeffs_f = args[3]
    condor_root = options.condor_root
    out_file = options.out_file

    run_cmd("mkdir -p %s" % condor_root)

    print "loading model..."
    with open(model_f, 'r') as f:
        model = cPickle.load(f)
    print "done."

    #print "loading standardized coefficients..."
    #with open(std_coeffs_f, 'r') as f:
    #    label_to_coeffs = json.load(f)
    #print "done"

    gene_names = model.classifier.feat_names
    label_to_classifier = model.classifier.label_to_classifier

    submit_f = join(condor_root, SUBMIT_FNAME)
    prepare_condor_environment(
        env_dir, 
        exp_list_name, 
        condor_root,
        submit_f,
        label_to_classifier, 
        gene_names  
    )
    run_condor_jobs(condor_root, SUBMIT_FNAME)
    gather_output_from_jobs(
        condor_root, 
        env_dir, 
        exp_list_name, 
        out_file
    )

def prepare_condor_environment(env_dir, exp_list_name, condor_root, submit_f, label_to_classifier, gene_names):

    submit_builder = condor_submit_tools.SubmitFileBuilder(
        EXECUTABLE_FNAME,
        20000,
        500,
        op_sys_version="7",
        blacklist=[ 
            "roy-1.biostat.wisc.edu",
            "lafleur.biostat.wisc.edu",
            "nebula-1.biostat.wisc.edu",
            "pulsar-84.biostat.wisc.edu"
        ]
    )
    run_cmd("rm -r %s" % condor_root)
    run_cmd("mkdir -p %s" % condor_root)
    run_cmd("ln -sf %s %s" % (SUBMIT_LOC, join(condor_root, EXECUTABLE_FNAME)))

    #for batch_id, batch_studies in enumerate(study_batches):
    for label, classif in label_to_classifier.iteritems():
        coeffs = classif.coef_[0]
        label_str = pdoo.convert_node_to_str(label).replace(":", "_")
        # prepare the jobs root directory
        job_root = join(condor_root, "%s.root" % label_str)
        label_data_f = join(job_root, "%s.json" % label_str)
        run_cmd("mkdir -p %s" % job_root)
        with open(label_data_f, 'w') as f:
            f.write(json.dumps({
                'gene_names': list(gene_names),
                'coefficients': list(coeffs),
                'label': pdoo.convert_node_to_str(label)
            }))
        in_files_locs = [
            THE_CONDOR_TARBALL,
            label_data_f
        ]
        symlinks = []
        for in_f in in_files_locs:
            symlink = join(job_root, basename(in_f))
            run_cmd("ln -sf %s %s" % (in_f, symlink))
            symlinks.append(symlink)

        # add to the job to the submit file
        submit_builder.add_job(
            job_root,
            arguments=[
                basename(label_data_f)
            ],
            input_files=in_files_locs,
            output_files=[PER_JOB_OUT_F]
        )
    with open(submit_f, 'w') as f:
        f.write(
            submit_builder.build_file()
        )




       

def gather_output_from_jobs(condor_root, env_dir, exp_list_name, out_f):
    pos_label_to_go_term_to_fdr = {}
    neg_label_to_go_term_to_fdr = {}
    for job_root in os.listdir(condor_root):
        if not isdir(join(condor_root, job_root)):
            continue
        job_out_f = join(condor_root, job_root, PER_JOB_OUT_F)
        print "Retrieving data from %s..." % job_out_f
        with open(job_out_f, 'r') as f:
            job_output = json.load(f)
        label = job_output['label']
        pos_go_term_to_fdr = job_output['pos_GO_term_to_FDR']
        neg_go_term_to_fdr = job_output['neg_GO_term_to_FDR']
        pos_label_to_go_term_to_fdr[label] = pos_go_term_to_fdr
        neg_label_to_go_term_to_fdr[label] = neg_go_term_to_fdr

    with open(out_f, 'w') as f:
        f.write(
            json.dumps(
                {
                    "label_to_pos_term_to_fdr": pos_label_to_go_term_to_fdr,
                    "label_to_neg_term_to_fdr": neg_label_to_go_term_to_fdr,
                    "data_set": {
                        "environment": env_dir,
                        "experiment_list_name": exp_list_name
                    }
                },
                indent=4
            )
        )



def run_condor_jobs(condor_root, submit_fname, cluster_id=None):
    print "Running Condor jobs..."
    cwd = os.getcwd()
    chdir(condor_root)
    if not cluster_id:
        num_jobs, cluster_id = condor_submit_tools.submit(submit_fname)
    jobs_still_going = True
    while jobs_still_going:
        print "Checking if any jobs are still running..."
        job_ids = query_condor_jobs.get_job_ids(cluster_id)
        print "Job ids returned by query: %s" % job_ids
        if len(job_ids) == 0: # No more jobs in this cluster
            jobs_still_going = False
        else:
            jobs_still_going = False
            for job_id in job_ids:
                #job_id = "%s.%s" % (str(cluster_id), str(i))
                status = query_condor_jobs.get_job_status_in_queue(job_id)
                if status != "H":
                    print "Found job %s with status %s. Will keep checking..." % (job_id, status)
                    jobs_still_going = True
                    break
            time.sleep(900)
    print "No jobs were found running. Finished."
    chdir(cwd)



def run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)

def chdir(loc):
    print "cd %s" % loc
    os.chdir(loc)

if __name__ == "__main__":
    main()
