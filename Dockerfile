# Start with an NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Define the working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Use noninteractive mode for apt-get to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools, dependencies, and Python 3.9
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3.9 -m venv .venv

# Activate virtual environment and install dependencies from requirements.txt
RUN . .venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Add and install cuDNN
COPY cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb /tmp
RUN dpkg -i /tmp/cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-8.9.5.29/cudnn-local-275FA572-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y libcudnn8 libcudnn8-dev && \
    rm /tmp/cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dlib with CUDA support
RUN git clone -b 'v19.22' --single-branch https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . && \
    cd .. && \
    /app/.venv/bin/python setup.py install && \
    cd /app && \
    rm -rf dlib

# Move .venv to a temporary location
RUN mv .venv /tmp/.venv_temp

# Set the default command
CMD ["/bin/bash"]
