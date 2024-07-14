# Classification with Noisy Labels

The repo is for the task of classification with noisy labels, based on the DivideMix model. The original code and paper can be found at [DivideMix GitHub](https://github.com/LiJunnan1992/DivideMix) and [DivideMix Paper](https://openreview.net/pdf?id=HJgExaVtwr).

## Repository Structure

- **.vscode**: Scripts for debugging.
- **data**: Stores the processed Wafer data.
  - **raw/MixedWM38.npz**: Original data from [WaferMap GitHub](https://github.com/Junliangwangdhu/WaferMap).
  - **Wafer/TransformedWaferMapData.npz**: Data processed in `src/data_preprocessing.ipynb` to convert one-hot encoding labels to numbers. Data split into train, validation, and test sets with ratios 0.7, 0.15, 0.15. The data is [Processed Data](https://drive.google.com/file/d/1-Ji7zNLlY1Uf3aVpQ0HyqrAfug4_ff_H/view?usp=sharing).
- **src**: Contains source code for the task.
  - **old_not_used**: Early stage code, kept for reference.
  - **pred_2nets**: The combined Move-to-unreliable strategy mentioned in the report where the safe set is built based on the prediction of two networks. If the label is not consistent with the set, the data will be moved to the unreliable set to help regularize the model.
  - **pred_2nets_drop**: The combined Drop strategy mentioned in the report where the safe set is built based on the prediction of two networks as well, but the data will be dropped if the label is not consistent with the set.
  - **pred_2nets_reverse**: Not reported in the report. The safe set is still built based on the prediction of two networks, but the data assessed this time is the unlabeled noisy data. If it is consistent with the set, the unreliable data will be moved to the reliable set. Preliminary experiments without tuning show that the method is not effective.
  - **pred_ind_net**: Not reported in the report. The safe set is built based on the prediction of individual networks, and if the label is not consistent with the set, the data will be moved to the unreliable set to help regularize the model. Preliminary experiments without tuning show that the method is not effective.
  - **pred_no_prestopping**: The original DivideMix with no prestopping utilized. Codes have been adapted but the basics are from the original DivideMix.

## Running the Models

- Update the environment location and activation in the run shell scripts:

  ```sh
  source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
  conda activate ProMix
  ```

  The code is built under Python 3.8. Important packages are listed in `requirements.txt`. Other packages can be installed as prompted. Download [Processed Data](https://drive.google.com/file/d/1-Ji7zNLlY1Uf3aVpQ0HyqrAfug4_ff_H/view?usp=sharing), and put best_model and checkpoint/safe_set.pkl to checkpoint folder

- To run the models from the very beginning (pre-early stop training and post-early stop training for the combined strategy), run:

  ```sh
  ./run_from_beginning.sh
  ```

  Default noise level is 50%.

- To run the models from the post-early stop training, ensure the checkpoint of the pred_2nets model and its corresponding last 10 epochs safe set are in the checkpoint folder. Then, simply run the following command:
  ```sh
  ./run_from_early_stop.sh
  ```
  Default noise level is 50%.

## Requirements
