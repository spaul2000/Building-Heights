#!/bin/bash

echo "Authenticating with Google Cloud..."
gcloud auth application-default login

if [ $? -ne 0 ]; then
    echo "Google Cloud authentication failed. Exiting."
    exit 1
fi

echo "Authenticated with Google Cloud successfully."

echo "Starting preprocessing..."
python code/preprocess/preprocess.py

if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Exiting."
    exit 1
fi

echo "Preprocessing completed successfully."

echo -n "Enter an experiment name for the training: "
read exp_name

echo "Starting training of height estimation model with experiment name: $exp_name"
python3 code/height_estimation.py train --exp_name "$exp_name"

if [ $? -ne 0 ]; then
    echo "Model training failed. Exiting."
    exit 1
fi

echo "Model training completed successfully."

echo -n "Enter the full path to the saved model checkpoint: "
read model_checkpoint_path

echo "Starting testing of height estimation model..."
python3 code/height_estimation.py test --ckpt_path "$model_checkpoint_path"

if [ $? -ne 0 ]; then
    echo "Model testing failed. Exiting."
    exit 1
fi

echo "Model testing completed successfully."
echo "Training pipeline completed successfully."