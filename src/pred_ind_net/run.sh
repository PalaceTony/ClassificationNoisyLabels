# For CIFAR-10-Symmetric

source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate ProMix

# export PYTHONPATH="/hpc2hdd/home/mgong081/Projects/ClassificationNoisy"
# export NCCL_SOCKET_IFNAME=eth2
export NOW=$(date -d '+8 hours' +"%m-%d_%H-%M-%S")
export DATE=$(date -d '+8 hours' +"%Y-%m-%d")
export TIME=$(date -d '+8 hours' +"%H-%M-%S")
export OUTPUT_DIR="outputs/debug/${DATE}/${TIME}"

# Save the current script to a temporary file
SCRIPT_FILE=$(mktemp)
cat "$0" > "$SCRIPT_FILE"
# Ensure the output directory exists
mkdir -p ${OUTPUT_DIR}
# Copy the temporary script file to the output directory
cp "$SCRIPT_FILE" "${OUTPUT_DIR}/run_script.sh"
# Clean up the temporary script file
rm "$SCRIPT_FILE"

python src/pred_ind_net/run.py \
"log_dir=${OUTPUT_DIR}" \
"robust_directly=True" \
"best_model_path=checkpoint/" \
# "best_model_name='model_epoch_35.pth'" \
# "load_model_during_first_train=True" \
