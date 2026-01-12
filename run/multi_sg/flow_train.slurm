#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sg-training
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/runs/sg/%x.out
#SBATCH --error=/scratch/bcse/jredepenning/runs/sg/%x.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=redep028@umn.edu

RUN_TYPE="sg"
export PROJECT_DIR=/projects/bcse/jredepenning/
export AMPLFI_CONTAINER_ROOT=/scratch/bcse/jredepenning/
export AMPLFI_OUTDIR=$PROJECT_DIR/amplfi-outdir/runs/${RUN_TYPE}
export AMPLFI_CONDORDIR=$AMPLFI_OUTDIR/condor

export AMPLFI_DATADIR=/projects/bcse/saggarwal/Data/online/
export FLOW_CONFIG=$PROJECT_DIR/amplfi/run/${RUN_TYPE}/sg.yaml

# launch training pipeline
apptainer run --nv -B ${AMPLFI_DATADIR} -B ${AMPLFI_OUTDIR} -B ${FLOW_CONFIG} \
        ${AMPLFI_CONTAINER_ROOT}/delta.sif \
        amplfi-flow-cli fit \
        --config ${FLOW_CONFIG}


## sg
