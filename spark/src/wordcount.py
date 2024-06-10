import os
import path
import configparser


from pyspark.sql import SparkSession
from operator import add

class WordCounter:
    def __init__(self, spark, filepath):
        self.spark = spark
        self.filepath = filepath

    def count_words(self):
        # Загрузка файла
        lines = self.spark.read.text(self.filepath).rdd.map(lambda r: r[0])

        # Разбивка на слова и подсчет
        word_counts = lines.flatMap(lambda x: x.split(" ")) \
                            .map(lambda x: (x, 1)) \
                            .reduceByKey(add)

        return word_counts.collect()

if __name__ == "__main__":
    # Создание SparkSession
    spark = SparkSession.builder \
        .appName("WordCount") \
        .getOrCreate()

    main_path = path.Path(__file__).absolute()
    main_path = main_path.parent.parent
    config = configparser.ConfigParser()
    config.read(os.path.join(main_path, 'config.ini'))
    # Путь к файлу
    filepath = os.path.join(main_path, config['data']['test_input'])

    # Создание экземпляра класса WordCounter
    word_counter = WordCounter(spark, filepath)

    # Выполнение подсчета слов
    results = word_counter.count_words()

    # Печать результатов
    for word, count in results:
        print("{}: {}".format(word, count))

    # Остановка SparkSession
    spark.stop()
