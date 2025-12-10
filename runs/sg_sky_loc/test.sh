#!/bin/bash

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
apptainer run --nv -B ${AMPLFI_DATADIR} -B ${AMPLFI_OUTDIR} -B ${FLOW_CONFIG} \
        ${AMPLFI_CONTAINER_ROOT}/delta.sif \
        amplfi-flow-cli test \
        --config ${FLOW_CONFIG} \
        --ckpt_path ${CKPT}

