# Flexpert-Design

In this directory we provide the code to train and run inference with Flexpert-Design. To expedite the release of the codebase, this part of code was not thoroughly curated and contains redundant files and code. The codebase might be revised in the future but probably it will get completely rewritten as part of a future project with an improved model.

## Environment

Tested for Python 3.9. For other versions enviroment might need to be adapted.

Assuming you have already installed the environment for Flexpert-3D and Flexpert-Seq, install the additional dependencies for Flexpert-Design using the `requirements.txt` file in this directory.

```bash
pip install -r requirements.txt
```


## Inference

In this example we will illustrate how to run inference with the trained model (trained wights are provided inside the train/results directory, you do not need to train the model again necessarily).

Place the PDB files you want to predict in the `predict_example` directory. It is expected that the files are named like `PDBCODE_CHAINID.pdb`, example file '1ahy_A.pdb' is provided. For each PDB file in that folder, add the instructions on flexibility you want to be considered by the ProteinMPNN model in the `PDBCODE_CHAINID_instructions.csv` file - example file '1ah7_A_instructions.csv' is provided. Then run the following command to run inference.

```bash
python3 predict.py \
    --infer_path predict_example/
```

The output will be in the `predict_example/predictions.txt` file.

The origininal sequence and the regenerated sequence can be compared using the following script.

```bash
python3 predict_example/compare_seqs.py \
    --pdb_code 1ah7_A
```

## Training

First make sure you have the Flexpert-3D model weights in the `Flexpert/models/weights` directory. Alternatively run the following script to download the weights.

```bash
. ../download_flexpert_weights.sh
```

Download the training data:

```bash
. ../download-cath-data.sh
```

Then run the following command to train the model.

```bash
export HF_HOME=./HF_cache
python3 train.py \
    --batch_size 4 \
    --model_name 'ProteinMPNN' \
    --stage 'fit' \
    --dataset FLEX_CATH4.3 \
    --ex_name training-reproduction \
    --offline 0 \
    --gpus 1 \
    --epoch 11 \
    --use_dynamics 1 \
    --flex_loss_coeff 0.8 \
    --init_flex_features 1 \
    --grad_normalization 0 \
    --loss_fn MSE \
    --use_pmpnn_checkpoint 1
```