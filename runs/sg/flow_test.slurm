#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=flow-testing
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/saggarwal/run1/%x.out
#SBATCH --error=/scratch/bcse/saggarwal/run1/%x.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=aggar104@umn.edu

export AMPLFI_CONTAINER_ROOT=/scratch/bcse/saggarwal/
export AMPLFI_DATADIR=/projects/bcse/saggarwal/data-LHO/online/
export AMPLFI_OUTDIR=/projects/bcse/saggarwal/amplfi-outdir/run1
export AMPLFI_CONDORDIR=$AMPLFI_OUTDIR/condor
export FLOW_CONFIG=/projects/bcse/saggarwal/amplfi/run1/cbc.yaml
export CKPT=/projects/bcse/saggarwal/amplfi-outdir/run1/train_logs/amplfi/4phsoagi/checkpoints/last.ckpt
export AMPLFI_TESTOUTDIR=/projects/bcse/saggarwal/amplfi-outdir/mdc

# launch training pipeline
apptainer run --nv -B ${AMPLFI_DATADIR} -B ${AMPLFI_OUTDIR} -B ${FLOW_CONFIG} -B ${SIM_CKPT}\
        ${AMPLFI_CONTAINER_ROOT}/delta.sif \
        amplfi-flow-cli test \
        --config ${FLOW_CONFIG} \
	--ckpt_path ${CKPT}
