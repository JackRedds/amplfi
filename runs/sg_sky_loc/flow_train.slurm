#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sg-sky-loc
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/sg_runs/sg_sky_loc/%x.out
#SBATCH --error=/scratch/bcse/jredepenning/sg_runs/sg_sky_loc/%x.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=redep028@umn.edu

RUN_TYPE="sg_sky_loc"
export SSL_CERT_FILE=/projects/bcse/jredepenning/amplfi/runs/sg_sky_loc/cacert.pem
export PROJECT_DIR=/projects/bcse/jredepenning/
export AMPLFI_CONTAINER_ROOT=/scratch/bcse/jredepenning/
export AMPLFI_OUTDIR=$PROJECT_DIR/amplfi-outdir/sg_runs/${RUN_TYPE}
export AMPLFI_CONDORDIR=$AMPLFI_OUTDIR/condor

export AMPLFI_DATADIR=/projects/bcse/jredepenning/gwak2_background
export FLOW_CONFIG=$PROJECT_DIR/amplfi/runs/${RUN_TYPE}/sg.yaml

# launch training pipeline
apptainer run --nv -B ${AMPLFI_DATADIR} -B ${AMPLFI_OUTDIR} -B ${FLOW_CONFIG} \
        ${AMPLFI_CONTAINER_ROOT}/delta_test.sif \
        amplfi-flow-cli fit \
        --config ${FLOW_CONFIG}


## sg
