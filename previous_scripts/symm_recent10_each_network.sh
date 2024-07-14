# For CIFAR-10-Symmetric

source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate ProMix

python src/run_recent10_each_network.py \
"robust_directly=True" \
"best_model_path=checkpoint/" \
# "best_model_name='model_epoch_35.pth'" \
# "load_model_during_first_train=True" \
