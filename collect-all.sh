#!/bin/bash
#SBATCH --partition=ANON REPLACE ME
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --array=4-5%7
#SBATCH --job-name=collect-all-results
#SBATCH --export=ALL

# NOTE: Right not all array id's are specified right now but feel free to edit as needed. 

set -e
set -u
set -o pipefail
set -x

# MLP Experiments with different splits
if [ "$SLURM_ARRAY_TASK_ID" == "1" ]; then
  python e1.py collect --expname "category3-nsplits-3" --fail-early --out mlp-nsplits-3.parquet
elif [ "$SLURM_ARRAY_TASK_ID" == "2" ]; then
  python e1.py collect --expname "category3-nsplits-5" --fail-early --out mlp-nsplits-5.parquet
elif [ "$SLURM_ARRAY_TASK_ID" == "3" ]; then
  python e1.py collect --expname "category3-nsplits-10" --fail-early --out mlp-nsplits-10.parquet

# RF Experiments with different splits
elif [ "$SLURM_ARRAY_TASK_ID" == "4" ]; then
  python e1.py collect --expname "category4-nsplits-3" --fail-early --out rf-nsplits-3.parquet
elif [ "$SLURM_ARRAY_TASK_ID" == "5" ]; then
  python e1.py collect --expname "category4-nsplits-5" --fail-early --out rf-nsplits-5.parquet
elif [ "$SLURM_ARRAY_TASK_ID" == "6" ]; then
  python e1.py collect --expname "category4-nsplits-10" --fail-early --out rf-nsplits-10.parquet

# Optimizer experiments with only 10 splits
elif [ "$SLURM_ARRAY_TASK_ID" == "7" ]; then
  python e1.py collect --expname "category5-nsplits-10" --fail-early --out optimizer-nsplits-10.parquet
else
  echo "OOB with slurm array task id $SLURM_ARRAY_TASK_ID"
fi
