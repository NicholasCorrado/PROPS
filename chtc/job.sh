#!/bin/bash

pid=$1  # ranges from 0 to num_commands*num_jobs-1 
step=$2 # ranges from 0 to num_jobs-1
echo $cmd $pid $step

# fetch your code from /staging/
CODENAME=PROPS
cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz

# Dcoker work-around
#python3 -m venv env
#source env/bin/activate
#cd PROPS
#pip install -r requirements.txt

cd PROPS
export PYTHONPATH=custom-envs:$PYTHONPATH # pip install -e fails on chtc because we don't have admin privileges .
#pip install --user -e custom-envs

# run your script -- $step ensures seeding is consistent across experiment batches
$($cmd --run_id $step --seed $step)

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
mv results_${pid}.tar.gz ..
