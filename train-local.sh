#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts performs local training for a TensorFlow model.

set -ev

echo "Training local ML model"



DATE=$(date '+%Y%m%d_%H%M%S')
MODEL_DIR=/tmp/trained_models/census_$DATE
PACKAGE_PATH=./trainer
TRAIN_DATA=./train_data
BATCH_SIZE=156
N_A=512
EMBEDDING_DIM=100
NUM_EPOCHS=1
TRAIN_EXAMPLES=950


gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train_data_path=${TRAIN_DATA} \
        --n_a=${N_A} \
        --batch_size=${BATCH_SIZE} \
        --embedding_dim=${EMBEDDING_DIM} \
        --num_epochs=${NUM_EPOCHS} \
        --train_examples=$TRAIN_EXAMPLES \
        --job-dir="${MODEL_DIR}"