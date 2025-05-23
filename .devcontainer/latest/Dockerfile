FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-24.04

# Install CUDA and prerequisites
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb

# Install libtinfo5 for CUDA, which is not available in the default Ubuntu 24.04 repo
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb && rm libtinfo5_6.3-2ubuntu0.1_amd64.deb

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gcc-14 g++-14 cuda-toolkit-12-4 doxygen graphviz

RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/vscode/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/vscode/.bashrc

# Install Catch2
WORKDIR /home/vscode
RUN git clone https://github.com/catchorg/Catch2.git
WORKDIR /home/vscode/Catch2
RUN cmake -Bbuild -H. -DBUILD_TESTING=OFF
RUN cmake --build build/ --target install --config Release
RUN cmake --build build/ --target install --config RelWithDebInfo
RUN cmake --build build/ --target install --config Debug
