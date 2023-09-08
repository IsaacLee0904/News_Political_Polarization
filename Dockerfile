FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install ckiptagger[tf,gdown] && \
    pip install bs4 && \
    pip install requests && \
    pip install scrapy && \
    pip install selenium && \
    pip install pandas && \
    pip install numpy && \
    pip install -U scikit-learn && \
    pip install gensim && \
    
CMD ["python", "main.py"]
