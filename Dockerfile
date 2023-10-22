FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y gcc g++ git && \
    conda create -n env python=3.8 && \
    echo "source activate env" > ~/.bashrc && \
    /bin/bash -c "source activate env && \
    pip install --upgrade pip && \
    pip install tensorflow==2.12.0 && \
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