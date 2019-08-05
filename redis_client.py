import redis


def set_model_version(model_version):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('model_version', model_version)


def get_model_version():
    r = redis.Redis(host='localhost', port=6379, db=0)
    ver = r.get('model_version')
    return ver.decode("utf-8")
