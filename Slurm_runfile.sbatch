#!/bin/bash -l
## Example run script for SLURM

## General configuration options
#SBATCH -J LcCOUP
#SBATCH -o myjob.%j.%N.out
#SBATCH -e myjob.%j.%N_Err.out
#SBATCH --mail-user=manuel.schmid@uni-tuebingen.de
#SBATCH --mail-type=ALL

## Machine and CPU configuration
## Number of tasks per job:
#SBATCH -n 1
## Number of nodes:
#SBATCH -N 1
## Define Node (I just use the newest...)


##Set the correct python PATH variables
#LANDLABWRKDIR=/esd/esd/data/mschmid/model_runs/runningSimus
LANDLABWRKDIR=$(pwd)
USERNAME=$(whoami)
PYTHONBIN=/esd/esd01/data/${USERNAME}/anaconda3/bin
RUNFILE=runfile_space.py

##Set up correct folder structure
#if [ ! -d DEM ] ; then mkdir DEM ; fi
#if [ ! -d ACC ] ; then mkdir ACC ; fi
#if [ ! -d DHDT ] ; then mkdir DHDT ; fi
#if [ ! -d NC ] ; then mkdir NC ; fi
#if [ ! -d SA ] ; then mkdir SA ; fi
#if [ ! -d dd ] ; then mkdir dd ; fi
#if [ ! -d CSVOutput ] ; then mkdir CSVOutput ; fi
#if [ ! -d SoilDepth ] ; then mkdir SoilDepth ; fi

##Run Landlab
${PYTHONBIN}/python ${LANDLABWRKDIR}/${RUNFILE}
