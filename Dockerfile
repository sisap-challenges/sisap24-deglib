# Define base image/operating system
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software
#RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates software-properties-common gpg-agent
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda3 -b
ENV PATH=/miniconda3/bin:${PATH}

SHELL ["/bin/bash", "--login", "-c"]
RUN conda init bash

#RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && apt install -y python3.12 python3-pip
RUN python3 -m pip install h5py tensorflow
RUN python3 -m pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple deglib==0.1.54

# Copy files and directory structure to working directory
COPY . . 
#COPY bashrc ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Run commands specified in "run.sh" to get started

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
#ENTRYPOINT [ "/bin/bash", "/sisap23-run.sh"]
