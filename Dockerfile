ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3

FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 libcurl4 curl libb64-dev
RUN pip install --upgrade pip
RUN pip install toposort networkx pytest nltk tqdm html2text progressbar
RUN pip --no-cache-dir --no-cache install git+https://github.com/NVIDIA/dllogger wget

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git

#Copy the perf_client over
ARG TRTIS_CLIENTS_URL=https://github.com/NVIDIA/triton-inference-server/releases/download/v2.2.0/v2.2.0_ubuntu1804.clients.tar.gz
RUN mkdir -p /workspace/install \
    && curl -L ${TRTIS_CLIENTS_URL} | tar xvz -C /workspace/install

#Install the python wheel with pip
RUN pip install /workspace/install/python/triton*.whl

WORKDIR /workspace/sebert
COPY . .

ENV PYTHONPATH /workspace/sebert
ENV BERT_PREP_WORKING_DIR /workspace/sebert/data
ENV PATH //workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
