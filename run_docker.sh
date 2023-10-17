#!/bin/bash

# Convert path to Windows format
win_path=$(pwd -W)
lowercase_path=$(echo $win_path | tr '[:upper:]' '[:lower:]' | sed 's|\\|/|g')

# Use the transformed path to run your Docker container, for example:
docker run --gpus all -it --rm -v "/$lowercase_path":/app face_recognition_gpu bash
