#!/bin/bash
# You need to modify this dataset path manually! 
DATASET_DIR=""

# Workspace. 
WORKSPACE=`pwd`

# Calculate features. 
python prepare_data.py calculate_features --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feat_type=logmel

# Pack features. 
python prepare_data.py pack_features --workspace=$WORKSPACE --feat_type=logmel

# Compute scaler. 
python prepare_data.py compute_scaler --workspace=$WORKSPACE --feat_type=logmel

# Train. 
python main_dnn.py train --workspace=$WORKSPACE --feat_type=logmel

# Inference. 
python main_dnn.py inference --workspace=$WORKSPACE --model_name=md_10000iters.tar --feat_type=logmel
