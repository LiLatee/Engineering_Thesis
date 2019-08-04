import redis


def set_model_version(model_version):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('model_version', model_version)
    print("REDIS model_version set to" + str(model_version))


def get_model_version():
    r = redis.Redis(host='localhost', port=6379, db=0)
    ver = r.get('model_version')
    print("REDIS model_version = " + str(ver))
    return ver.decode("utf-8")
