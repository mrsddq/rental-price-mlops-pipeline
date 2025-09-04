FROM python

LABEL maintainer="laraibks@gmail.com"

WORKDIR /mlapp

COPY main.py main.py
COPY requirements.txt requirements.txt

# RUN pip install numpy pandas scikit-learn
RUN pip install -r requirements.txt

CMD ["python","main.py"]
