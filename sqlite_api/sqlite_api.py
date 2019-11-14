from flask import Flask, request, jsonify
# from client_cass import CassandraClient
from client_SQLite import DatabaseSQLite
import json
import pickle

app = Flask(__name__)
# cass = CassandraClient()
sqlite = DatabaseSQLite()


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/samples', methods=['GET'])
def get_all_samples_as_list_of_dicts():
    return jsonify(sqlite.get_all_samples_as_list_of_dicts())


@app.route('/samples-for-update', methods=['GET'])
def get_samples_for_update_model():
    id = request.args.get('last_sample_id')
    return jsonify(sqlite.get_samples_to_update_model_as_list_of_dicts(id))


@app.route('/last_sample_id', methods=['GET'])
def get_last_sample_id():
    return str(sqlite.get_last_sample_id())


# @app.route('/samples', methods=['DELETE'])
# def delete_all_samples():
#     sqlite.delete_all_samples()
#     return 'deleted'

@app.route('/samples', methods=['POST'])
def insert_sample():
    sample_dict = request.json
    sqlite.insert_sample_as_dict(sample_dict)
    return 'sample inserted'

@app.route('/models', methods=['POST'])
def insert_ModelInfo():
    model_info = pickle.loads(request.data)
    sqlite.insert_ModelInfo(model_info) # todo trzeba sprawdzić
    return 'model inserted'

@app.route('/models/get_last', methods=['GET'])
def get_last_ModelInfo():
    return pickle.dumps(sqlite.get_last_ModelInfo()) # todo trzeba sprawdzić

@app.route('/models/', methods=['GET'])
def get_last_version_of_specified_model():
    model_name = request.args.get("model_name")
    return pickle.dumps(sqlite.get_last_version_of_specified_model(model_name))

@app.route('/models/get_as_list_of_dicts', methods=['GET'])
def get_all_models_history_as_list_of_dicts():
    return pickle.dumps(sqlite.get_all_models_history_as_list_of_dicts())

@app.route('/models/get_as_list_of_ModelInfo', methods=['GET'])
def get_all_models_history_as_list_of_ModelInfo():
    return pickle.dumps(sqlite.get_all_models_history_as_list_of_ModelInfo())

@app.route('/models/get_id_of_last_specified_model/', methods=['GET'])
def get_id_of_last_specified_model():
    model_name = request.args.get("model_name")
    return str(sqlite.get_id_of_last_specified_model(model_name))

@app.route('/restart',)
def restart():
    sqlite.restart_cassandra()
    return 'Cassandra restarted'

if __name__  == "__main__":
    app.run(host='0.0.0.0', port=8764)