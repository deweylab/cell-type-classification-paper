root=/ua/mnbernstein/projects/tbcp/cello_dev
basename=condor_tarball
out_dir=$root/train_test_experiments/leave_study_out_condorized/$basename

rm -r $out_dir
mkdir -p $out_dir

cp $root/train_model.py $out_dir
cp $root/train_model.py $out_dir

cp $root/train_test_experiments/leave_study_out_condorized/condor_job.py condor_tarball
cp -r $root/common condor_tarball
cp -r $root/models condor_tarball
cp -r $root/training_parameter_sets condor_tarball

tar -zcf $basename.tar.gz $basename
