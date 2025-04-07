#!/bin/bash

pid=$1  # ranges from 0 to num_commands*num_jobs-1 
step=$2 # ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
cmd="${cmd} --run_id ${step} --seed ${step}"
echo $cmd

# fetch your code from /staging/
CODENAME=PROPS
cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz

cd PROPS
export PYTHONPATH=custom-envs:$PYTHONPATH # pip install -e fails on chtc because we don't have admin privileges .
mkdir gymnasium_local
python -m pip install 'gymnasium[mujoco]==0.29.1' --target gymnasium_local
python -m pip install 'mujoco==2.3.5' --target mujoco_local
export PYTHONPATH=gymnasium_local:$PYTHONPATH
export PYTHONPATH=mujoco_local:$PYTHONPATH
$cmd

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
mv results_${pid}.tar.gz ..
