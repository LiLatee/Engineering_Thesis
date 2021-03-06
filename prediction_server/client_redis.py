import redis

class DatabaseRedis:
    def __init__(self, model_id):
        self.model_id = model_id
        self.redis = redis.StrictRedis(host='redis_service', port=6379, db=0)
        # self.redis = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)

    def rpush_sample(self, json_sample):
        self.redis.rpush('samples_model_' + str(self.model_id), json_sample)

    def get_all_samples_as_list_of_bytes(self):
        result_list = self.redis.lrange('samples_model_' + str(self.model_id), 0, -1)
        self.redis.ltrim('samples_model_' + str(self.model_id), len(result_list), -1)
        return result_list

    def del_all_samples(self):
        self.redis.delete('samples_model_' + str(self.model_id))

    def key_exists(self, model_id):
        return self.redis.exists('samples_mode_' + str(model_id))


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
