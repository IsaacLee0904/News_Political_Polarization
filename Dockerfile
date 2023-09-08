FROM tensorflow/tensorflow:2.5.0-gpu

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install ckiptagger[tf,gdown]==0.2.1 && \
    pip install bs4==0.0.1 && \
    pip install requests==2.25.1 && \
    pip install scrapy==2.5.0 && \
    pip install selenium==3.141.0 && \
    pip install pandas && \
    pip install numpy && \
    pip install scikit-learn && \
    pip install gensim==4.0.1 && \
    pip install python-Levenshtein && \
    pip install urllib3==1.26.6 && \
    pip install chardet==4.0.0
 
CMD ["python", "main.py"]
