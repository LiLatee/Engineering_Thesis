import redis


def set_model_version(model_version):
    r = redis.Redis(host='redis_service', port=6379, db=0)
    r.set('model_version', model_version)
