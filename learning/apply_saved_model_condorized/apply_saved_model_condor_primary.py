#########################################################################
#   Run leave-study-out cross validation. The folds are batched and
#   used as jobs for Condor
#########################################################################

import sys
import os
from os.path import join, basename, isdir
import math
import time

from optparse import OptionParser
import json
import collections
from collections import defaultdict, Counter
import numpy as np
import subprocess

sys.path.append("/ua/mnbernstein/projects/tbcp/data_management/recount2")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/learning")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")
sys.path.append("/ua/mnbernstein/projects/condor_tools")

import condor_submit_tools
import query_condor_jobs
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from machine_learning import learners

THE_CONDOR_TARBALL = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/apply_saved_model_condorized/apply_saved_model_condor_tarball.tar.gz"
SUBMIT_FNAME = "condor.submit" 
EXECUTABLE_FNAME = "apply_saved_model_condor_executable.bash"
SUBMIT_LOC = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/apply_saved_model_condorized/apply_saved_model_condor_executable.bash"
PER_JOB_OUT_F = "apply_saved_model_results_batch.json"


def main():
    usage = "usage: %prog <config file> <environment dir> <experiment list name>"
    parser = OptionParser(usage)
    parser.add_option("-o", "--out_file", help="File to write experiment results")
    parser.add_option("-c", "--condor_root", help="The directory in Condor places output")
    (options, args) = parser.parse_args()

    model_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    n_exps_per_job = int(args[3])
    condor_root = options.condor_root
    out_f = options.out_file

    exp_list_f = join(
        env_dir,
        'data/experiment_lists',
        exp_list_name,
        'experiment_list.json'
    )
    with open(exp_list_f, 'r') as f:
        the_exps = json.load(f)['experiments']

    # split the studies in batches to give to each job
    # for now, each study gets its own job
    exp_batches = [
        x
        for x in chunks(
            the_exps,
            n_exps_per_job
        )
    ]
    submit_f = join(condor_root, SUBMIT_FNAME)
    #prepare_condor_environment(env_dir, exp_list_name, condor_root, submit_f, exp_batches, model_f)
    #run_condor_jobs(condor_root, submit_f)
    gather_output_from_jobs(condor_root, env_dir, exp_list_name, out_f)


def prepare_condor_environment(env_dir, exp_list_name, condor_root, submit_f, exp_batches, model_f):
    # generate the submit file
    submit_builder = condor_submit_tools.SubmitFileBuilder(
        EXECUTABLE_FNAME,
        15000,
        500,
        op_sys_version="7",
        blacklist=[ # TODO These might change!
            "roy-1.biostat.wisc.edu",
            "lafleur.biostat.wisc.edu",
            "nebula-1.biostat.wisc.edu",
            "pulsar-84.biostat.wisc.edu"
        ]
    )
    run_cmd("rm -r %s" % condor_root) 
    run_cmd("mkdir -p %s" % condor_root)
    run_cmd("ln -sf %s %s" % (SUBMIT_LOC, join(condor_root, EXECUTABLE_FNAME)))

    for batch_id, batch_exps in enumerate(exp_batches):
        # prepare the jobs root directory
        job_root = join(condor_root, "%d.root" % batch_id)
        exps_list_f = join(job_root, "%d.json" % batch_id)
        run_cmd("mkdir -p %s" % job_root)
        with open(exps_list_f, 'w') as f:
            f.write(json.dumps(list(batch_exps)))
        in_files_locs = [
            THE_CONDOR_TARBALL,
            exps_list_f,
            model_f
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
                basename(model_f),
                env_dir,
                exp_list_name,
                basename(exps_list_f)
            ],
            input_files=in_files_locs,
            output_files=[PER_JOB_OUT_F]
        )
    with open(submit_f, 'w') as f:
        f.write(
            submit_builder.build_file()
        )



def gather_output_from_jobs(condor_root, env_dir, exp_list_name, out_f):
    exp_to_predictions = {}
    for job_root in os.listdir(condor_root):
        if not isdir(join(condor_root, job_root)):
            continue
        job_out_f = join(condor_root, job_root, PER_JOB_OUT_F)
        print "Retrieving data from %s..." % job_out_f
        with open(job_out_f, 'r') as f:
            e_to_p = json.load(f)
        exp_to_predictions.update(e_to_p)
    with open(out_f, 'w') as f:
        f.write(
            json.dumps(
                {
                    "predictions": exp_to_predictions,
                    "data_set": {
                        "environment": env_dir,
                        "experiment_list_name": exp_list_name
                    }
                },
                indent=4,
                separators=(',', ': ')
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




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n+1):
        yield l[i:i + n + 1]


def run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)

def chdir(loc):
    print "cd %s" % loc
    os.chdir(loc)

if __name__ == "__main__":
    main()
