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
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from machine_learning import learners
import leave_study_out as lso
import query_condor_jobs

THE_CONDOR_TARBALL = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/leave_study_out_cv_condor_tarball.tar.gz"
SUBMIT_FNAME = "condor.submit" 
EXECUTABLE_FNAME = "leave_study_out_cv_condor_executable.bash"
SUBMIT_LOC = "/ua/mnbernstein/projects/tbcp/phenotyping/learning/leave_study_out_cv_condor_executable.bash"
PER_JOB_OUT_F = "leave_study_out_cv_results_batch.json"

DEFAULT_N_JOBS = 20

MAX_EXPS_IN_JOB = 50

def main():
    usage = "usage: %prog <config file> <environment dir> <experiment list name>"
    parser = OptionParser(usage)
    parser.add_option("-o", "--out_file", help="File to write experiment results")
    parser.add_option("-r", "--artifacts_parent_dir", help="The directory in which to write temporary files")
    (options, args) = parser.parse_args()

    config_f = args[0]
    env_dir = args[1]
    exp_list_name = args[2]
    out_f = options.out_file
    artifacts_parent_dir = options.artifacts_parent_dir

    exp_list_f = join(
        env_dir,
        "data",
        "experiment_lists",
        exp_list_name,
        "experiment_list.json"
    )
    data_set_metadata_f = join(
        env_dir,
        "data",
        "data_set_metadata.json"
    )
    with open(exp_list_f, 'r') as f:
        the_exps = set(json.load(f)['experiments'])
    with open(data_set_metadata_f, 'r') as f:
        exp_to_info = json.load(f)
    study_to_exps = defaultdict(lambda: [])
    for exp, info in exp_to_info.iteritems():
        if not exp in the_exps:
            continue
        study = info['study_accession']
        study_to_exps[study].append(exp)

    # split the studies in batches to give to each job
    # for now, each study gets its own job
    the_studies_l = sorted(study_to_exps.keys())
    """
    study_batches = [
        [study]
        for study in the_studies_l
    ]
    """
    #study_batches = [
    #    x
    #    for x in chunks(
    #        the_studies_l,
    #        int(math.ceil(len(the_studies_l) / num_procs))
    #    )
    #]
    study_batches = []
    for study in the_studies_l:
        study_exps = study_to_exps[study]
        print "Study %s. Num exps: %d" % (study, len(study_exps))
        if len(study_exps) > MAX_EXPS_IN_JOB:
            exp_batches = chunks(study_exps, MAX_EXPS_IN_JOB)
            for exp_batch in exp_batches:
                study_batches.append([{
                    "study": study,
                    "experiments": exp_batch
                }])
        else:
            study_batches.append([{
                "study": study,
                "experiments": study_to_exps[study]
            }])

    condor_root = join(artifacts_parent_dir, "condor_root")
    submit_f = join(condor_root, SUBMIT_FNAME)
    prepare_condor_environment(config_f, env_dir, exp_list_name, condor_root, submit_f, study_batches)
    run_condor_jobs(condor_root, submit_f)
    gather_output_from_jobs(condor_root, env_dir, exp_list_name, out_f)


def prepare_condor_environment(config_f, env_dir, exp_list_name, condor_root, submit_f, study_batches):
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
            "pulsar-84.biostat.wisc.edu",
            "pulsar-9.biostat.wisc.edu"
        ]
    )
    run_cmd("rm -r %s" % condor_root) 
    run_cmd("mkdir -p %s" % condor_root)
    run_cmd("ln -sf %s %s" % (SUBMIT_LOC, join(condor_root, EXECUTABLE_FNAME)))

    for batch_id, batch_studies in enumerate(study_batches):
        # prepare the jobs root directory
        job_root = join(condor_root, "%d.root" % batch_id)
        studies_list_f = join(job_root, "%d.json" % batch_id)
        run_cmd("mkdir -p %s" % job_root)
        with open(studies_list_f, 'w') as f:
            f.write(json.dumps(list(batch_studies)))
        in_files_locs = [
            THE_CONDOR_TARBALL,
            studies_list_f,
            config_f
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
                basename(config_f),
                env_dir,
                exp_list_name,
                basename(studies_list_f)
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
            job_output = json.load(f)
        e_to_p = job_output['predictions']
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
