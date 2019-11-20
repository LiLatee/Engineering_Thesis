FROM python:3-onbuild

#FROM python:3.7-alpine
COPY . /usr/src/app
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt
#RUN apt-get install python3.7
#ENV PATH=$PATH:"../Engineering_Thesis"
#ENV PYTHONPATH "../Engineering_Thesis"
CMD ["python", "cassandra_api.py"]
#ADD restart_cassandra.py /
#CMD ["python", "./restart_cassandra.py"]

#COPY /home/marcin/PycharmProjects/Engineering_Thesis/cassandra_api/restart_cassandra.py /
#ENTRYPOINT ["python restart_cassandra.py"]

