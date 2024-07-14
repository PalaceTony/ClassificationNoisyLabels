source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate ProMix

export NOW=$(date -d '+8 hours' +"%m-%d_%H-%M-%S")
export DATE=$(date -d '+8 hours' +"%Y-%m-%d")
export TIME=$(date -d '+8 hours' +"%H-%M-%S")
export BASE_OUTPUT_DIR="outputs/debug/${DATE}/${TIME}"

# Save the current script to a temporary file
SCRIPT_FILE=$(mktemp)
cat "$0" > "$SCRIPT_FILE"
# Ensure the base output directory exists
mkdir -p ${BASE_OUTPUT_DIR}

# Run the script for different values of 'r'
for r_value in 0.2; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/r_${r_value}"
    mkdir -p ${OUTPUT_DIR}
    cp "$SCRIPT_FILE" "${OUTPUT_DIR}/run_script.sh"
    
    python src/pred_no_prestopping/run.py \
    "log_dir=${OUTPUT_DIR}" \
    "r=${r_value}" \
    "num_epochs=100"
done

# Clean up the temporary script file
rm "$SCRIPT_FILE"