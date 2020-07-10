#########################################################################
#   Run leave-study-out cross validation. Parallelize with Condor where
#   each held-out study is run as a Condor job
#########################################################################

import os
from os.path import join, basename, isdir
import time
from optparse import OptionParser
import json
from collections import defaultdict
import subprocess

from common import load_dataset
from condor_tools import condor_submit_tools
from condor_tools import query_condor_jobs

SUBMIT_FNAME = "condor.submit" 
EXECUTABLE_FNAME = "condor_executable.bash"
OUT_FILENAME = "fold_classification_results.tsv"
BLACKLIST = [
    "roy-1.biostat.wisc.edu",
    "lafleur.biostat.wisc.edu",
    "nebula-1.biostat.wisc.edu",
    "nebula-7.biostat.wisc.edu",
    "nebula-8.biostat.wisc.edu",
    "broman-8.biostat.wisc.edu"
]

def main():
    usage = ""
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    config_f = args[0]
    dataset_dir = args[1]
    condor_root = args[2]
    condor_tarball_f = args[3]
    condor_exec_f = args[4]

    # Load the study-to-experiment mapping
    dummy_feats = 'log_tpm' # These don't matter
    r = load_dataset.load_dataset(dataset_dir, dummy_feats)
    study_to_exps = r[8]

    # Create a fold file for each study that will tell each job
    # the set of experiments to hold out
    folds = []
    for study, exps in study_to_exps.items():
        exps = study_to_exps[study]
        print("Creating fold for study {} with {} samples.".format(
            study, 
            len(exps)
        ))
        folds.append({
            "study": study,
            "experiments": sorted(exps)
        })

    # The Condor submit file
    submit_f = join(condor_root, SUBMIT_FNAME)

    # Prepare Condor's root directory and run 
    # Condor
    prepare_condor_environment(
        dataset_dir,
        condor_root,
        config_f, 
        submit_f,
        condor_tarball_f, 
        condor_exec_f,
        folds
    )
    run_condor_jobs(condor_root, submit_f)


def prepare_condor_environment(
        dataset_dir,
        condor_root,
        config_f,
        submit_f, 
        condor_tarball_f,
        condor_exec_f,
        folds
    ):
    # Generate the submit file
    submit_builder = condor_submit_tools.SubmitFileBuilder(
        EXECUTABLE_FNAME,
        15000,
        500,
        op_sys_version="7",
        blacklist=BLACKLIST
    )

    # Setup the Condor root directory
    run_cmd("rm -r %s" % condor_root) 
    run_cmd("mkdir -p %s" % condor_root)
    run_cmd("ln -sf %s %s" % (
        condor_exec_f, 
        join(condor_root, basename(condor_exec_f))
    ))

    for fold_i, fold in enumerate(folds):
        # Setup the job's root directory
        job_root = join(condor_root, "fold_%d.root" % fold_i)
        fold_f = join(job_root, "fold_%d.json" % fold_i)
        run_cmd("mkdir -p %s" % job_root)
        with open(fold_f, 'w') as f:
            f.write(json.dumps(fold))

        # The set of input files for the current job
        in_files_locs = [
            condor_tarball_f,
            fold_f,
            config_f
        ]
        
        # Create symlinks in each job's directory to the 
        # job's input files
        for in_f in in_files_locs:
            symlink = join(job_root, basename(in_f))
            run_cmd("ln -sf %s %s" % (in_f, symlink))

        # Add job to the submit file
        submit_builder.add_job(
            job_root,
            arguments=[
                basename(config_f),
                dataset_dir,
                basename(fold_f),
                OUT_FILENAME
            ],
            input_files=in_files_locs,
            output_files=[OUT_FILENAME]
        )
    with open(join(condor_root, submit_f), 'w') as f:
        f.write(
            submit_builder.build_file()
        )



def gather_output_from_jobs(condor_root, env_dir, exp_list_name, out_f):
    exp_to_predictions = {}
    for job_root in os.listdir(condor_root):
        if not isdir(join(condor_root, job_root)):
            continue
        job_out_f = join(condor_root, job_root, OUT_FILENAME)
        print("Retrieving data from {}...".format(job_out_f))
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
    print("Running Condor jobs...")
    cwd = os.getcwd()
    chdir(condor_root)
    if not cluster_id:
        num_jobs, cluster_id = condor_submit_tools.submit(submit_fname)
    return
    jobs_still_going = True
    while jobs_still_going:
        print("Checking if any jobs are still running...")
        job_ids = query_condor_jobs.get_job_ids(cluster_id)
        print("Job ids returned by query: {}").format(job_ids)
        if len(job_ids) == 0: # No more jobs in this cluster
            jobs_still_going = False
        else:
            jobs_still_going = False
            for job_id in job_ids:
                #job_id = "%s.%s" % (str(cluster_id), str(i))
                status = query_condor_jobs.get_job_status_in_queue(job_id)
                if status != "H":
                    print("Found job %s with status {}. Will keep checking...".format(job_id, status))
                    jobs_still_going = True
                    break
            time.sleep(900)
    print("No jobs were found running. Finished.")
    chdir(cwd)



def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True, env=None)

def chdir(loc):
    print("cd ", loc)
    os.chdir(loc)

if __name__ == "__main__":
    main()
