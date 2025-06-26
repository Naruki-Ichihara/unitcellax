FROM nvcr.io/nvidia/jax:25.04-py3

# Build argument to control Claude Code installation
ARG INSTALL_CLAUDE=false

RUN apt update
RUN apt upgrade -y
RUN pip install --upgrade pip
RUN pip install numpy==1.26.0
#RUN apt install libxinerama1 -y
#RUN apt install libxcursor1 -y
#RUN apt install libglu1-mesa -y
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:fenics-packages/fenics
RUN apt update

# Install Node.js and Claude Code (if enabled)
RUN if [ "$INSTALL_CLAUDE" = "true" ]; then \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs && \
    npm install -g @anthropic-ai/claude-code; \
    fi

# TODO: Installing fenicsx in dockerfile breaks MPI setup. We need to find a way to install fenicsx without breaking MPI. 
# Please install fenicsx manually in the container.

#RUN pip install fenics-basix
#RUN apt install -y fenicsx

WORKDIR /home/
CMD ["/bin/bash"]
