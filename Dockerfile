FROM tensorflow/tensorflow:2.8.0-gpu

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y wget

RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    wget -qO - https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add -

RUN apt-get install -y gcc g++ git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh

RUN ~/miniconda/bin/conda create -n env python=3.8

WORKDIR /app

COPY . /app

RUN /bin/bash -c "source $HOME/miniconda/bin/activate env && \
    pip install --upgrade pip && \
    pip install ckiptagger[tf,gdown]==0.2.1 && \
    pip install ckip-transformers && \
    pip install bs4==0.0.1 && \
    pip install requests==2.25.1 && \
    pip install scrapy==2.5.0 && \
    pip install selenium==3.141.0 && \
    pip install pandas && \
    pip install numpy && \
    pip install matplotlib && \
    pip install scikit-learn && \
    pip install gensim==4.0.1 && \
    pip install python-Levenshtein && \
    pip install urllib3==1.26.6 && \
    pip install networkx && \
    pip install chardet==4.0.0"

CMD ["python", "main.py"]
