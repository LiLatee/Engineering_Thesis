import redis


class DatabaseRedis:
    def __init__(self):
        self.redis = redis.StrictRedis(host='redis_service', port=6379, db=0)

    def rpush_sample(self, json_sample):
        self.redis.rpush('samples', json_sample)

    def get_all_samples_as_list_of_bytes(self):
        queue = self.redis.lrange('samples', 0, -1)
        self.redis.ltrim('samples', len(queue), -1)
        return queue

    def del_all_samples(self):
        self.redis.delete('samples')
