#! /bin/bash

. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_environ_var.sh
. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_secrets.sh

echo $HF_HOME

singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/coseg.sif jupyter notebook --no-browser --ip='0.0.0.0' --port=8080

# In your lcoal terminal
# ssh -L 8080:chenguang01.engr.wustl.edu:8080 jianhong.t@ssh.seas.wustl.edu
# jupyter notebook --no-browser --ip='0.0.0.0' --port=8080

