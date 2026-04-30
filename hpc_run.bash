#!/usr/bin/bash

# don't define anything before the PBS options
# don't put any directly comments behind the PBS options

#PBS -m eba
#PBS -M jannis.ruh@student.uts.edu.au
#PBS -N nongeneric_free_fermions

# 200h is the maximum, otherwise the job doesn't even get queued
#PBS -l walltime=2:00:00 
# see for max possible resource on a single node: https://hpc.research.uts.edu.au/status/
# (select=1 is probably the default (putting stuff onto one chunk(/host?)))
#PBS -l select=1:ncpus=10:mem=8GB

# this is relative to the final workdir which is ./=${PBS_O_WORKDIR}, so we don't have
# to move it from the scratch
#PBS -e ./log/
#PBS -o ./log/


bin="src/main.py"

cd ${PBS_O_WORKDIR}
mkdir -p log
mkdir -p output

scratch="/scratch/${USER}_${PBS_JOBID%.*}"
mkdir -p ${scratch}/output
cp -r src ${scratch}
cp apptainer.sif ${scratch}

cd ${scratch}

apptainer exec apptainer.sif ./${bin}
# NOTE: `cd ${PBS_O_WORKDIR}; mv ${scratch}/output/* output` doesn't work; it's the wild
# card that makes problems in this case, but I don't know why (maybe the ${scratch} name
# is too weird)?
mv output/* ${PBS_O_WORKDIR}/output/

cd ${PBS_O_WORKDIR}
rm -rf ${scratch}
