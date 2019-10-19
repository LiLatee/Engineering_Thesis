import redis

# TODO TO jest potrzebne????
def get_model_version():
    r = redis.Redis(host='redis_service', port=6379, db=0)
    ver = r.get('model_version')
    return ver.decode("utf-8")
