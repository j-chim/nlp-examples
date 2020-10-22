FROM continuumio/miniconda3

RUN apt-get update \
    && apt-get install -y --reinstall build-essential \
    && apt-get install -y gcc git git-lfs 

ENV SYFERTEXT_BRANCH "syfertext_0.1.0"
ENV HOME_DIR "/data"
COPY . ${HOME_DIR}
WORKDIR ${HOME_DIR}

# PySyft
RUN git clone https://github.com/OpenMined/PySyft.git \
    && cd PySyft \
    && python setup.py install 

RUN cd ${HOME_DIR}

# SyferText
RUN git lfs install \
    && pip install git+git://github.com/Nilanshrajput/syfertext_en_core_web_lg@master \
    && git clone -b ${SYFERTEXT_BRANCH} https://github.com/j-chim/SyferText.git \
    && cd SyferText \
    && python setup.py install 

EXPOSE 8888