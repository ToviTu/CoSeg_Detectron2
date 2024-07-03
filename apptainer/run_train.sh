#! /bin/bash
. /home/research/jianhong.t/CoSeg_Detectron2/apptainer/set_environ_var.sh
echo $HF_HOME

singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/coseg.sif bash $WORKING_DIR/run.sh

