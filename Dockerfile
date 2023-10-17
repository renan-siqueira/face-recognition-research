# Start with an NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Define the working directory inside the container
WORKDIR /app

# Install basic tools, dependencies, and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python libraries
RUN pip3 install opencv-python-headless numpy face_recognition

# Copy the project files to the container
COPY . .

# Set the default command
# CMD ["python3.11", "nome_do_seu_script.py"]
CMD ["/bin/bash"]
