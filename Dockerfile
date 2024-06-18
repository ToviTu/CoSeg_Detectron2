# Start from a PyTorch image with CUDA
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

# Set the working directory in the container
WORKDIR /app
RUN mkdir /dataset/
RUN mkdir /models/
RUN apt-get -y update; apt-get install -y --no-install-recommends \
    wget \
    curl \
    dnsutils \
    nano \
    zip \
    unzip \
    git \
    s3cmd \
    ffmpeg \
    screen \
    fonts-freefont-ttf \
    inotify-tools \
    parallel \
    pciutils \
    ncdu \
    libbz2-dev \
    gettext \
    apt-transport-https \
    gnupg2 \
    time \
    openssl \
    redis-tools \
    ca-certificates \
    hdf5-tools\
    vim\
    build-essential 

# Copy the dependencies file to the working directory
COPY . .

# Install any dependencies
RUN pip install -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
