#! /bin/bash

. /home/research/jianhong.t/CoSeg_Detectron2/apptainer/set_environ_var.sh
singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/coseg.sif /bin/bash
echo $HF_HOME
