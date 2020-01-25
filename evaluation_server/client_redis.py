import redis

class DatabaseRedis:
    def __init__(self, model_id):
        self.model_id = model_id
        self.redis = redis.StrictRedis(host='redis_service', port=6379, db=0)
        self.del_all_samples()

        self.num_processed_samples = 0
        self.correct_predictions = 0
        self.avg_acc = 0
        self.avg_aucroc = 0
        self.avg_f1_score = 0
        self.num_of_evaluations = 0

    def rpush_sample(self, json_sample):
        self.redis.rpush('samples_model_' + str(self.model_id), json_sample)

    def get_all_samples_as_list_of_json(self):
        result_list = self.redis.lrange('samples_model_' + str(self.model_id), 0, -1)
        self.redis.ltrim('samples_model_' + str(self.model_id), len(result_list), -1)
        return result_list

    def del_all_samples(self):
        self.redis.delete('samples_model_' + str(self.model_id))

    def key_exists(self):
        return self.redis.exists('samples_model_' + str(self.model_id))


if __name__ == '__main__':
    db = DatabaseRedis(model_id=1)
    db.rpush_sample("hejjjj")
    print(db.key_exists(1))

    # d = {"a": 1, "b": 2, "c": 3}
    # db.rpush_sample(json.dumps(d))
    # d = {"a": 11, "b": 22, "c": 33}
    # db.rpush_sample(json.dumps(d))
    # result = db.get_all_samples_as_list_of_bytes()
    # print(result)
    # db.del_all_samples()
