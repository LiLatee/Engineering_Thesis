# how to open cqlsh:
# 1. docker exec -it <container_with_cassandra> bash
# 2. cqlsh

from cassandra.cluster import Cluster
from typing import List, Dict, Union
from cassandra.query import dict_factory

RowAsDictType = Dict[str, Union[str, float, int]]


class CassandraClient:

    def __init__(self) -> None:
        super().__init__()
        self.KEYSPACE = 'keyspace_name'

        cluster = Cluster(['cassandra_api'], port=9042)

        is_exception = True
        while is_exception:
            try:
                self.session = cluster.connect()
                self.session.row_factory = dict_factory
            except:
                is_exception = True
                continue
            is_exception = False

        self.delete_keyspace_if_exists()
        self.create_keyspace(self.session)
        self.session.set_keyspace(self.KEYSPACE)


    def create_keyspace(self, session) :
        session.execute("""
        CREATE KEYSPACE IF NOT EXISTS """ + self.KEYSPACE + """
        WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' }
        """)

    def delete_keyspace_if_exists(self):
        self.session.execute('DROP KEYSPACE IF EXISTS ' + self.KEYSPACE)



if __name__ == '__main__':
    db = CassandraClient()