#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ubuntu:22.04

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user || true
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN chmod -R 777 /home

RUN echo "Installing the base dependencies..."
RUN DEBIAN_FRONTEND=noninteractive apt update && apt upgrade -y && apt install -y --no-install-recommends \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  wget \
  curl \
  vim \
  nano \
  python3 \
  python3-pip \
  python3-venv \
  tmux \
  nvtop \
  htop \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /workdir
WORKDIR /workdir
RUN echo "Installing conda..."
RUN curl "https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh" > /workdir/conda_install.sh
RUN bash /workdir/conda_install.sh -p /workdir/conda -b
RUN rm /workdir/conda_install.sh
ENV PATH=/workdir/conda/bin:${PATH}
#RUN echo "Setting up certificates so git will work properly"
#RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y --no-install-recommends --reinstall ca-certificates && apt clean && rm -rf /var/lib/apt/lists/*

COPY docker_entry_script.sh /workdir/docker_entry_script.sh
RUN chmod ugo+rx /workdir/docker_entry_script.sh
USER user
RUN conda init
RUN conda create --name relaqs
RUN echo "source /workdir/docker_entry_script.sh" >> ~/.bashrc


#ENTRYPOINT ["/workdir/docker_entry_script.sh"]


#RUN git clone git@github.com:akataba/rl-repo.git
#RUN python3 -m venv /workdir/venv
#ENV PATH="/workdir/venv/bin:$PATH"

