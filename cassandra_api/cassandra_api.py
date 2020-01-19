from flask import Flask, request
from client_cass import CassandraClient
import json

app = Flask(__name__)
cass = CassandraClient()


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/samples', methods=['GET'])
def get_all_samples_as_list_of_dicts():
    return str(cass.get_all_samples_as_list_of_dicts())


@app.route('/samples-for-update', methods=['GET'])
def get_samples_for_update_model():
    id = request.args.get('last_sample_id')
    return json.dumps(cass.get_samples_for_update_model_as_list_of_dicts(id))


@app.route('/last_sample_id', methods=['GET'])
def get_last_sample_id():
    return str(cass.get_last_sample_id())


@app.route('/samples', methods=['DELETE'])
def delete_all_samples():
    cass.delete_all_samples()
    return 'deleted'

@app.route('/samples', methods=['POST'])
def insert_sample():
    sample_dict = request.json
    cass.insert_sample_as_dict(sample_dict)
    return 'inserted'

@app.route('/restart',)
def restart():
    cass.restart_cassandra()
    return 'Cassandra restarted'

if __name__  == "__main__":
    app.run(host='0.0.0.0', port=8764)