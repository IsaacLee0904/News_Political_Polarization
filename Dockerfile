FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install ckiptagger[tf,gdown]==0.2.1 && \
    pip install bs4==0.0.1 && \
    pip install requests==2.25.1 && \
    pip install scrapy==2.5.0 && \
    pip install selenium==3.141.0 && \
    pip install pandas==1.3.3 && \
    pip install numpy==1.21.2 && \
    pip install scikit-learn==0.24.2 && \
    pip install gensim==4.0.1 && \
    pip install urllib3==1.26.6 && \
    pip install chardet==4.0.0
 
CMD ["python", "main.py"]
