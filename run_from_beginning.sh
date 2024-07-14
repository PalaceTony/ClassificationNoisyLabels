#!/bin/bash

source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate ProMix

# export PYTHONPATH="/hpc2hdd/home/mgong081/Projects/ClassificationNoisy"
# export NCCL_SOCKET_IFNAME=eth2
export NOW=$(date -d '+8 hours' +"%m-%d_%H-%M-%S")
export DATE=$(date -d '+8 hours' +"%Y-%m-%d")
export TIME=$(date -d '+8 hours' +"%H-%M-%S")
export OUTPUT_DIR="outputs/${DATE}/${TIME}"

# Save the current script to a temporary file
SCRIPT_FILE=$(mktemp)
cat "$0" > "$SCRIPT_FILE"
# Ensure the output directory exists
mkdir -p ${OUTPUT_DIR}
# Copy the temporary script file to the output directory
cp "$SCRIPT_FILE" "${OUTPUT_DIR}/run_script.sh"
# Clean up the temporary script file
rm "$SCRIPT_FILE"

# Specify the model folder here
MODEL_FOLDER="pred_2nets_drop" # Available model folders: pred_2nets, pred_2nets_drop, pred_2nets_reverse, pred_ind_net, pred_no_prestopping

python src/${MODEL_FOLDER}/run.py \
"log_dir=${OUTPUT_DIR}" \