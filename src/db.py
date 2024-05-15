import os
import mysql.connector
import pandas as pd
from typing import Dict
import numpy as np


class Database():
    def __init__(self, spark, host="0.0.0.0", port=55000, database="lab6_bd"):
        self.username = "root"
        self.password = "0000"
        self.spark = spark
        self.client = mysql.connector.connect(
                                    user=self.username,
                                    password=self.password,
                                    database=database,
                                    host=host,
                                    port=port)
        self.jdbcUrl = f"jdbc:mysql://{host}:{port}/{database}"

    def read_table(self, tablename: str):
        return self.spark.read \
            .format("jdbc") \
            .option("url", self.jdbcUrl) \
            .option("user", self.username) \
            .option("password", self.password) \
            .option("dbtable", tablename) \
            .option("inferSchema", "true") \
            .load()

    def insert_df(self, df, tablename):
        df.write \
            .format("jdbc") \
            .option("url", self.jdbcUrl) \
            .option("user", self.username) \
            .option("password", self.password) \
            .option("dbtable", tablename) \
            .mode("append") \
            .save()

    def execute_query(self, query):
        try:
            with self.client.cursor() as cursor:
                cursor.execute(query)
            self.client.commit()
            print("Query executed successfully!")
        except Exception as e:
            print(f"Error executing query: {e}")

    def create_table(self, table_name: str, columns: Dict):
        print(f"Creating table {table_name}")
        cols = ", ".join([f"`{k}` {v}" for k, v in columns.items()])
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} 
            (
                {cols}
            )
            ENGINE = InnoDB;
        """
        self.execute_query(query)

    def insert_data(self, table_name: str, df):
        columns = ", ".join([f"`{col}`" for col in df.columns])
        placeholders = ", ".join(["%s" for _ in range(len(df.columns))])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        try:
            with self.client.cursor() as cursor:
                # Преобразуем DataFrame в список кортежей
                data = [tuple(row) for row in df.to_numpy()]
                # Вставляем данные в таблицу
                cursor.executemany(query, data)
            self.client.commit()
            print("Data inserted successfully!")
        except Exception as e:
            print(f"Error inserting data: {e}")

