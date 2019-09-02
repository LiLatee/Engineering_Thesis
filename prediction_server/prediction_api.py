from flask import Flask, request
from MLModel import MLModel
import json
from client_SQLite import DatabaseSQLite
app = Flask(__name__)
model = MLModel()

# @app.route('/')
# def hello_world():
#     return str(cass.get_model_history_all())
#     # return 'Hello, World!'
#
#
# @app.route('/restart')
# def restart():
#     cass.restart_cassandra()
#     return 'Cassandra restarted'
#
#
# @app.route('/sample')
# def sample():
#     return str(cass.get_all_samples_as_list_of_dicts())
#     # return str(len(get_sample_all()))


@app.route('/fit', methods=['POST'])
def fit():
    model.fit(request.data)
    return ''


@app.route('/predict', methods=['POST'])
def predict():
    # result = model.predict(request.data)
    # print("Result=" + str(result))
    return ''

from adapter_sqlite import AdapterDB
# starts updating a model
@app.route('/update_start', methods=['GET'])
def update_start():
    # model.update_start(json.loads(request.data))
    db = AdapterDB()
    sample_dict = json.loads('{"sale":0,"sales_amount_in_euro":-1.0,"time_delay_for_conversion":-1,"click_timestamp":1596439471,"nb_clicks_1week":0,"product_price":0.0,"product_age_group":"4C90FD52FC53D2C1C205844CB69575AB","device_type":"D7D1FB49049702BF6338894757E0D959","audience_id":"865A11AEC419E0B83637E01693CA6534","product_gender":"A5D15FC386510762EC0DDFF54ABE6F94","product_brand":"DC31D4641EBB3213EDA758F50F18482A","product_category_1":"7F286560861764CCB93C90B7AA833949","product_category_2":"1F7C0436C367A14B4C62294F235D45A4","product_category_3":"925C269ABCB1ABD1B1404E9AF9BAC32C","product_category_4":"-1","product_category_5":"-1","product_category_6":"-1","product_category_7":-1,"product_country":"2AC62132FBCFA093B9426894A4BC6278","product_id":"E0B8A58942AD00411F94DC480E187E64","product_title":"79FB52BC086F410AFFD19B69426DF923 7FE78A23011394B8883E77F532C225C9 35A837ED63F3C1176F1B22920AB7BEE1 330BB99FE6CAE65A7DA7F17B449156FB","partner_id":"DE8AF4DB950D7A4F844B744FA402C4A8","user_id":"95C3CF40CFEFD9DEF99995CD90A23D42"}')
    db.insert_sample(sample_dict=sample_dict)
    db.insert_sample(sample_dict=sample_dict)
    db.insert_sample(sample_dict=sample_dict)
    print(db.get_last_sample_id())
    print(db.get_samples_for_update_model_as_list_of_dicts(last_sample_id=2))
    return str(db.get_last_sample_id())
    # fr =str(db.get_all_models_history_as_df())
    # return fr
    # return ''


# sends signal to model, that updated model is ready and models should be replaced
@app.route('/update_ready', methods=['GET'])
def update_ready():
    model.update_ready()
    return ''


if __name__  == "__main__":
    app.run(host='0.0.0.0', port=5000)



