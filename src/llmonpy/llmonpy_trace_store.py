#   Copyright © 2024 Thomas Edward Burns
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import io
import json
import os
import sqlite3
from queue import Queue

JSON_STRING_COLUMN_NAME = "json_string"
TRACE_ID_COLUMN_NAME = "trace_id"
TRACE_GROUP_ID_COLUMN_NAME = "trace_group_id"
VARIATION_OF_TRACE_ID_COLUMN_NAME = "variation_of_trace_id"
TITLE_COLUMN_NAME = "title"
EVENT_ID_COLUMN_NAME = "event_id"
STEP_ID_COLUMN_NAME = "step_id"
STEP_NAME_COLUMN_NAME = "step_name"
TOURNEY_RESULT_ID_COLUMN_NAME = "tourney_result_id"
START_TIME_COLUMN_NAME = "start_time"


class LLMonPyConnectionPool:
    def __init__(self, database_path, max_connections=10):
        self.database_path = database_path
        self.max_connections = max_connections
        self.connection_list = []
        self.available_connections = Queue()
        for i in range(max_connections):
            connection = sqlite3.connect(database_path, check_same_thread=False)
            self.connection_list.append(connection)
            self.available_connections.put(connection)

    def get_connection(self):
        result = self.available_connections.get()
        return result

    def return_connection(self, connection):
        self.available_connections.put(connection)

    def dispose(self):
        for connection in self.connection_list:
            connection.close()

    class ConnectionContextManager:
        def __init__(self, pool):
            self.pool = pool
            self.connection = None

        def __enter__(self):
            self.connection = self.pool.get_connection()
            return self.connection

        def __exit__(self, exception_type, exception_value, exception_traceback):
            self.pool.return_connection(self.connection)

    def acquire(self):
        return self.ConnectionContextManager(self)


class QueryCondition:
    def __init__(self, column_name, operator, value):
        self.column_name = column_name
        self.operator = operator
        self.value = value

    def to_sql(self):
        return self.column_name + " " + self.operator + " \"" + self.value + "\""


class JSONTableColumn:
    def __init__(self, name, indexed=True, unique=False, column_type="TEXT"):
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
            with self.connection_pool.acquire() as connection:
                cursor = connection.cursor()
                statement = io.StringIO()
                statement.write("CREATE TABLE " + self.table_name + " (")
                first_column = True
                columns_to_index = []
                for column in self.column_list:
                    if first_column is False:
                        statement.write(", ")
                    statement.write(column.name + " " + column.column_type)
                    if first_column:
                        statement.write(" PRIMARY KEY")
                    else:
                        if column.unique:
                            statement.write(" UNIQUE")
                        elif column.indexed:
                            statement.write(" NOT NULL")
                            columns_to_index.append(column.name)

                    first_column = False
                statement.write(")")
                cursor.execute(statement.getvalue())
                connection.commit()
                for column_name in columns_to_index:
                    cursor.execute("CREATE INDEX " + self.table_name + "_" + column_name + "_index ON " + self.table_name + "(" + column_name + ")")
                    connection.commit()

    def table_exists(self):
        with self.connection_pool.acquire() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
            result = cursor.fetchone()
        return result is not None

    def insert_rows(self, object_list):
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
                statement.write(",?")
            else:
                statement.write("?")
            first_column = False
        statement.write(")")
        row_list = []
        for obj in object_list:
            row = []
            for column in self.column_list:
                if column.name == JSON_STRING_COLUMN_NAME:
                    row.append(obj.to_json())
                else:
                    row.append(getattr(obj, column.name))
            row_list.append(row)
        statement_text = statement.getvalue()
        with self.connection_pool.acquire() as connection:
            connection.executemany(statement_text, row_list)
            connection.commit()

    def get_distinct_values(self, column_name):
        statement = io.StringIO()
        statement.write("SELECT DISTINCT " + column_name + " FROM " + self.table_name)
        statement_text = statement.getvalue()
        with self.connection_pool.acquire() as connection:
            query_result = connection.execute(statement_text)
            result = []
            for row in query_result.fetchall():
                result.append(row[0])
        return result

    def select_rows(self, condition_list, object_factory, condition_operator="AND"):
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
        with self.connection_pool.acquire() as connection:
            query_result = connection.execute(statement_text)
            result = []
            for row in query_result.fetchall():
                dictionary = json.loads(row[0])
                value = object_factory(dictionary)
                result.append(value)
        return result

    def get_all(self, object_factory):
        statement = io.StringIO()
        statement.write("SELECT " + JSON_STRING_COLUMN_NAME + " FROM " + self.table_name)
        statement_text = statement.getvalue()
        with self.connection_pool.acquire() as connection:
            query_result = connection.execute(statement_text)
            result = []
            for row in query_result.fetchall():
                dictionary = json.loads(row[0])
                value = object_factory(dictionary)
                result.append(value)
        return result


class SqliteLLMonPyTraceStore:
    def __init__(self, data_directory, trace_factory, step_record_factory, event_factory, tourney_result_factory):
        self.data_directory = data_directory
        db_path = os.path.join(self.data_directory + '/trace_store.db')
        self.connection_pool = LLMonPyConnectionPool(db_path)
        self.trace_factory = trace_factory
        self.step_record_factory = step_record_factory
        self.event_factory = event_factory
        self.tourney_result_factory = tourney_result_factory
        self.trace_list_table = None
        self.step_record_table = None
        self.event_table = None
        self.tourney_result_table = None
        self.create_tables()

    def stop(self):
        self.connection_pool.dispose()

    def create_tables(self):
        trace_list_table_column_list = [JSONTableColumn(TRACE_ID_COLUMN_NAME, True, True),
                                        JSONTableColumn(TRACE_GROUP_ID_COLUMN_NAME, True),
                                        JSONTableColumn(VARIATION_OF_TRACE_ID_COLUMN_NAME, True),
                                        JSONTableColumn(START_TIME_COLUMN_NAME, False)]
        self.trace_list_table = JSONTable(self.connection_pool, "trace", trace_list_table_column_list,
                                          self.trace_factory)
        self.trace_list_table.create_table()
        step_record_table_column_list = [JSONTableColumn(STEP_ID_COLUMN_NAME, True, True),
                                        JSONTableColumn(TRACE_ID_COLUMN_NAME, True, False),
                                        JSONTableColumn(TRACE_GROUP_ID_COLUMN_NAME, True, False)]
        self.step_record_table = JSONTable(self.connection_pool, "step_record", step_record_table_column_list,
                                           self.step_record_factory)
        self.step_record_table.create_table()
        event_table_column_list = [JSONTableColumn(EVENT_ID_COLUMN_NAME, True, True),
                                   JSONTableColumn(TRACE_ID_COLUMN_NAME, True),
                                   JSONTableColumn(STEP_ID_COLUMN_NAME, True)]
        self.event_table = JSONTable(self.connection_pool, "event", event_table_column_list,
                                     self.event_factory)
        self.event_table.create_table()
        tourney_result_table_column_list = [JSONTableColumn(TOURNEY_RESULT_ID_COLUMN_NAME, True, True),
                                            JSONTableColumn(STEP_ID_COLUMN_NAME, True, False),
                                            JSONTableColumn(TRACE_ID_COLUMN_NAME, True, False),
                                            JSONTableColumn(STEP_NAME_COLUMN_NAME, True, False)]
        self.tourney_result_table = JSONTable(self.connection_pool, "tourney_result",
                                              tourney_result_table_column_list,
                                              self.tourney_result_factory)
        self.tourney_result_table.create_table()

    def insert_trace_info(self, trace_list):
        self.trace_list_table.insert_rows(trace_list)

    def insert_step_records(self, step_record_list):
        self.step_record_table.insert_rows(step_record_list)

    def insert_events(self, event_list):
        self.event_table.insert_rows(event_list)

    def insert_tourney_results(self, tourney_result_list):
        self.tourney_result_table.insert_rows(tourney_result_list)

    def get_trace_list(self):
        result = self.trace_list_table.get_all(self.trace_factory)
        return result

    def get_trace_by_id(self, trace_id):
        trace_id_condition = QueryCondition(TRACE_ID_COLUMN_NAME, "=", trace_id)
        record_list = self.trace_list_table.select_rows([trace_id_condition], self.trace_factory)
        result = record_list[0] if record_list is not None and len(record_list) > 0 else None
        return result

    def get_steps_for_trace(self, trace_id):
        trace_id_condition = QueryCondition(TRACE_ID_COLUMN_NAME, "=", trace_id)
        step_list = self.step_record_table.select_rows([trace_id_condition], self.step_record_factory)
        return step_list

    def get_events_for_trace(self, trace_id):
        trace_id_condition = QueryCondition(TRACE_ID_COLUMN_NAME, "=", trace_id)
        event_list = self.event_table.select_rows([trace_id_condition], self.event_factory)
        return event_list

    def get_events_for_step(self, step_id):
        step_id_condition = QueryCondition(STEP_ID_COLUMN_NAME, "=", step_id)
        event_list = self.event_table.select_rows([step_id_condition], self.event_factory)
        return event_list

    def get_tourney_results_for_trace(self, trace_id):
        trace_id_condition = QueryCondition(TRACE_ID_COLUMN_NAME, "=", trace_id)
        tourney_list = self.tourney_result_table.select_rows([trace_id_condition], self.tourney_result_factory)
        return tourney_list

    def get_tourney_step_name_list(self):
        result = self.tourney_result_table.get_distinct_values(STEP_NAME_COLUMN_NAME)
        return result

    def get_tourney_results_for_step_name(self, step_name):
        step_name_condition = QueryCondition(STEP_NAME_COLUMN_NAME, "=", step_name)
        tourney_list = self.tourney_result_table.select_rows([step_name_condition], self.tourney_result_factory)
        return tourney_list
