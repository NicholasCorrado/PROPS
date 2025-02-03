#!/bin/bash

pid=$1  # ranges from 0 to num_commands*num_jobs-1 
step=$2 # ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
echo $cmd $pid $step

# fetch your code from /staging/
CODENAME=DataAugmentationForRL
cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz
echo $CODENAME

# fetch conda env from stathing and activate it
#ENVNAME=da4rl
#ENVDIR=${ENVNAME}
#cp /staging/ncorrado/${ENVNAME}.tar.gz .
#mkdir $ENVDIR
#tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
#rm ${ENVNAME}.tar.gz # remove env tarball
#source $ENVDIR/bin/activate
#conda list

python3 -m venv env
source env/bin/activate

# install editable packages (editable packages cannot be packaged in the env tarball)
cd $CODENAME
pip install -e .
pip install -e src/custom-envs
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium panda-gym tyro pyyaml stable-baselines3 tensorboard gymnasium_robotics rliable
cd src

# run your script -- $step ensures seeding is consistent across experiment batches
$($cmd --run_id $step --seed $step)

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
mv results_${pid}.tar.gz ../..
