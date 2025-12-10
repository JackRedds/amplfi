RUN_TYPE="sg_sky_loc"
export PROJECT_DIR=/projects/bcse/jredepenning/
export AMPLFI_CONTAINER_ROOT=/scratch/bcse/jredepenning/
export AMPLFI_OUTDIR=$PROJECT_DIR/amplfi-outdir/run5/${RUN_TYPE}
export AMPLFI_CONDORDIR=$AMPLFI_OUTDIR/condor

export AMPLFI_DATADIR=$PROJECT_DIR/amplfi-data-dir-O3b-HLV_small
export FLOW_CONFIG=$PROJECT_DIR/amplfi/runs/${RUN_TYPE}/sg.yaml

# launch training pipeline
source /projects/bcse/jredepenning/amplfi/.venv/bin/activate
python tuing.py

## sg
