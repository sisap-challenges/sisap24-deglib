# Define base image/operating system
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software
#RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && apt install python3.12 -y
RUN python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deglib==0.1.51 && python3 -m pip install h5py tensorflow

# Copy files and directory structure to working directory
COPY . . 
#COPY bashrc ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Run commands specified in "run.sh" to get started

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
#ENTRYPOINT [ "/bin/bash", "/sisap23-run.sh"]
