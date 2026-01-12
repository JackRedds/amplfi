#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sg-tuning
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/run4/sg/%x.out
#SBATCH --error=/scratch/bcse/jredepenning/run4/sg/%x.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --mail-type=END
#SBATCH --mail-user=redep028@umn.edu

RUN_TYPE="multi_sg"
export PROJECT_DIR=/projects/bcse/jredepenning/
export AMPLFI_CONTAINER_ROOT=/scratch/bcse/jredepenning/
export AMPLFI_OUTDIR=$PROJECT_DIR/amplfi-outdir/sg_runs/${RUN_TYPE}
export AMPLFI_CONDORDIR=$AMPLFI_OUTDIR/condor

# export AMPLFI_DATADIR=/projects/bcse/deep1018/amplfi-data-dir-O3
export AMPLFI_DATADIR=/projects/bcse/jredepenning/gwak2_background
export FLOW_CONFIG=$PROJECT_DIR/amplfi/runs/${RUN_TYPE}/sg.yaml

# launch training pipeline
source /projects/bcse/jredepenning/amplfi/.venv/bin/activate
python tuing.py

## sg
