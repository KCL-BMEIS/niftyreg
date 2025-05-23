FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-20.04

# Install CMake from source
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.18.0"
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Install CUDA and prerequisites
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends cuda-11-8 doxygen graphviz

RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/vscode/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/vscode/.bashrc

# Install Catch2
WORKDIR /home/vscode
RUN git clone -b v3.7.1 https://github.com/catchorg/Catch2.git
WORKDIR /home/vscode/Catch2
RUN cmake -Bbuild -H. -DBUILD_TESTING=OFF
RUN cmake --build build/ --target install --config Release
RUN cmake --build build/ --target install --config RelWithDebInfo
RUN cmake --build build/ --target install --config Debug
