#   Copyright © 2024 Thomas Edward Burns
#  #
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#  #
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#  #
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#
import io
import json
import os
import sqlite3

from sqlalchemy import create_engine, text, MetaData, Column, Table, Index, String

JSON_STRING_COLUMN_NAME = "json_string"
TRACE_ID_COLUMN_NAME = "trace_id"
TRACE_GROUP_ID_COLUMN_NAME = "trace_group_id"
VARIATION_OF_TRACE_ID_COLUMN_NAME = "variation_of_trace_id"
TITLE_COLUMN_NAME = "title"
EVENT_ID_COLUMN_NAME = "event_id"
STEP_ID_COLUMN_NAME = "step_id"


class QueryCondition:
    def __init__(self, column_name, operator, value):
        self.column_name = column_name
        self.operator = operator
        self.value = value

    def to_sql(self):
        return self.column_name + " " + self.operator + " \"" + self.value + "\""


class JSONTableColumn:
    def __init__(self, name, indexed=True, unique=False, column_type=String):
        self.name = name
        self.indexed = indexed
        self.unique = unique
        self.column_type = column_type


class JSONTable:
    def __init__(self, connection_pool, table_name, search_column_list, object_factory):
        self.table_name = table_name
        self.column_list = search_column_list if search_column_list is not None else []
        self.column_list.append(JSONTableColumn(JSON_STRING_COLUMN_NAME, False))
        self.connection_pool = connection_pool
        self.object_factory = object_factory

    def create_table(self):
        if self.table_exists() is False:
            metadata = MetaData()

            columns = []
            for i, column in enumerate(self.column_list):
                col = Column(column.name, column.column_type)
                if column.indexed:
                    col = col.nullable(False)
                if i == 0:  # Set primary key for the first column
                    col = col.primary_key()
                columns.append(col)

            table = Table(self.table_name, metadata, *columns)
            metadata.create_all(self.connection_pool)

            with self.connection_pool.connect() as connection:
                for i, column in enumerate(self.column_list):
                    if column.indexed and i != 0:
                        index_name = f"{self.table_name}_{column.name}_index"
                        index = Index(index_name, table.c[column.name])
                        index.create(bind=self.connection_pool)

    def table_exists(self):
        with self.connection_pool.connect() as connection:
            query_result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                                             {"table_name": self.table_name})
            result = query_result.fetchone()
        return result is not None

    def insert_rows(self, object_list):
        with self.connection_pool.connect() as connection:
            statement = io.StringIO()
            statement.write("INSERT INTO " + self.table_name + " (")
            first_column = True
            for column in self.column_list:
                if first_column is False:
                    statement.write(", ")
                statement.write(column.name)
                first_column = False
            statement.write(") VALUES (")
            first_column = True
            for column in self.column_list:
                if first_column is False:
                    statement.write(",:")
                else:
                    statement.write(":")
                statement.write(column.name)
                first_column = False
            statement.write(")")
            row_list = []
            for obj in object_list:
                row = {}
                for column in self.column_list:
                    if column.name == JSON_STRING_COLUMN_NAME:
                        row[column.name] = obj.to_json()
                    else:
                        row[column.name] = getattr(obj, column.name)
                row_list.append(row)
            statement_text = statement.getvalue()
            with connection.begin() as transaction:
                connection.execute(text(statement_text), row_list)

    def select_rows(self, condition_list, object_factory, condition_operator="AND"):
        with self.connection_pool.connect() as connection:
            statement = io.StringIO()
            statement.write("SELECT " + JSON_STRING_COLUMN_NAME + " FROM " + self.table_name)
            if len(condition_list) > 0:
                statement.write(" WHERE ")
                first_condition = True
                for condition in condition_list:
                    if first_condition is False:
                        statement.write(" " + condition_operator + " ")
                    statement.write(condition.to_sql())
                    first_condition = False
            statement_text = statement.getvalue()
            try:
                query_result = connection.execute(text(statement_text))
                result = []
                for row in query_result.fetchall():
                    dictionary = json.loads(row[0])
                    value = object_factory(dictionary)
                    result.append(value)
            except Exception as e:
                print(str(e))
                raise e
        return result


class SqliteLLMonPyTraceStore:
    def __init__(self, data_directory, trace_factory, step_record_factory, event_factory):
        self.data_directory = data_directory
        db_path = os.path.join(self.data_directory + '/trace_store.db')
        self.connection_pool = create_engine(
            'sqlite:///' + str(db_path),
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
        self.trace_factory = trace_factory
        self.step_record_factory = step_record_factory
        self.event_factory = event_factory
        self.trace_list_table = None
        self.step_record_table = None
        self.event_table = None
        self.create_tables()

    def stop(self):
        self.connection_pool.dispose()

    def create_tables(self):
        trace_list_table_column_list = [JSONTableColumn(TRACE_ID_COLUMN_NAME, True, True),
                                        JSONTableColumn(TRACE_GROUP_ID_COLUMN_NAME, True),
                                        JSONTableColumn(VARIATION_OF_TRACE_ID_COLUMN_NAME, True),
                                        JSONTableColumn(TITLE_COLUMN_NAME, False)]
        self.trace_list_table = JSONTable(self.connection_pool, "trace", trace_list_table_column_list,
                                          self.trace_factory)
        self.trace_list_table.create_table()
        step_record_table_column_list = [JSONTableColumn(TRACE_ID_COLUMN_NAME, True),
                                         JSONTableColumn(TRACE_GROUP_ID_COLUMN_NAME, True)]
        self.step_record_table = JSONTable(self.connection_pool, "step_record", step_record_table_column_list,
                                           self.step_record_factory)
        self.step_record_table.create_table()
        event_table_column_list = [JSONTableColumn(EVENT_ID_COLUMN_NAME, True, True),
                                   JSONTableColumn(TRACE_ID_COLUMN_NAME, True),
                                   JSONTableColumn(STEP_ID_COLUMN_NAME, True)]
        self.event_table = JSONTable(self.connection_pool, "event", event_table_column_list,
                                     self.event_factory)
        self.event_table.create_table()

    def insert_trace(self, trace):
        self.trace_list_table.insert_rows([trace])

    def insert_step_records(self, step_record_list):
        self.step_record_table.insert_rows(step_record_list)

    def get_steps_for_trace(self, trace_id):
        trace_id_condition = QueryCondition(TRACE_ID_COLUMN_NAME, "=", trace_id)
        record_list = self.step_record_table.select_rows([trace_id_condition], self.step_record_factory)
        return record_list

    def insert_events(self, event_list):
        self.event_table.insert_rows(event_list)





