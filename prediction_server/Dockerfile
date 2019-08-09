FROM python:3-onbuild
COPY . /usr/src/app
#RUN pip install --upgrade pip
#COPY . /app/prediction_server/
#WORKDIR /app/prediction_server/
#RUN pip install scikit-learn
#RUN pip install -r /app/prediction_server/requirements.txt
CMD ["python", "prediction_api.py"]