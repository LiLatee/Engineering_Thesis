import redis
from typing import List, Union, Any


class RedisClient:
    def __init__(self):
        self.redis = redis.StrictRedis(host='redis_service', port=6379, db=0)

    def rpush_sample(self, json_sample: Union[str, Any]) -> None:
        self.redis.rpush('samples', json_sample)

    def get_all_samples_as_list_of_bytes(self) -> List:
        queue = self.redis.lrange('samples', 0, -1)
        self.redis.ltrim('samples', len(queue), -1)
        return queue

    def del_all_samples(self) -> None:
        self.redis.delete('samples')
