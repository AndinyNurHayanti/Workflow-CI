FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install mlflow
RUN pip install xgboost

CMD ["python","modelling.py"]