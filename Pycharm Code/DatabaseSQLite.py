from sqlalchemy import create_engine


class DatabaseSQLite:
    def __init__(self, dataframe):
        self.engine = create_engine('sqlite://', echo=False)
        dataframe[0:0].to_sql('samples', con=self.engine)

    def add_new_samples(self, dataframe):
        dataframe.to_sql('samples', con=self.engine, if_exists='append')
        number_of_rows, _ = dataframe.shape
        print("LOG: " + str(number_of_rows) + " new samples have been added")

    def clear_database(self):
        self.engine.execute("DELETE * FROM samples")
        print("LOG: " + "database has been cleared")