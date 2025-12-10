#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=flow-testing
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/run1/%x.out
#SBATCH --error=/scratch/bcse/jredepenning/run1/%x.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
export CKPT=$AMPLFI_OUTDIR/train_logs/best.ckpt

# launch training pipeline
apptainer run --nv -B ${AMPLFI_DATADIR} -B ${AMPLFI_OUTDIR} -B ${FLOW_CONFIG} -B ${CKPT} \
        ${AMPLFI_CONTAINER_ROOT}/delta_test.sif \
        amplfi-flow-cli test \
        --config ${FLOW_CONFIG} \
        --ckpt_path ${CKPT}
