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