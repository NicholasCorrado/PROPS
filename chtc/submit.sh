# make a temp commands file where spaces are replaced with * so that multi-word commands are properly parsed by the queue command
commands_file=commands/${1}.txt
commands_file_tmp=commands/${1}_tmp.txt
sed 's/ /*/g' "$commands_file" > "$commands_file_tmp"

# make directory to store results and logs
results_dir=results/${1}
mkdir -p ${results_dir}
mkdir -p ${results_dir}/logs

condor_submit job.sub \
  results_dir=${results_dir} \
  commands_file=${commands_file} \
  num_jobs=${2:-1}

# remove temp commands file
rm $commands_file_tmp
