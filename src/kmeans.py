import sys
import os
import path
import configparser

import numpy
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from src.preprocess import Preprocess


class KMeansModel:
    def clustering(self, final_data):
        silhouette_score = []

        evaluator = ClusteringEvaluator(predictionCol='prediction',
                                        featuresCol='scaled_features',
                                        metricName='silhouette',
                                        distanceMeasure='squaredEuclidean')

        for i in range(2, 10):
            kmeans = KMeans(featuresCol='scaled_features', k=i)
            model = kmeans.fit(final_data)
            predictions = model.transform(final_data)
            score = evaluator.evaluate(predictions)
            silhouette_score.append(score)
            print('Silhouette Score for k =', i, 'is', score)


def main():
    main_path = path.Path(__file__).absolute()
    main_path = main_path.parent.parent
    config = configparser.ConfigParser()
    config.read(os.path.join(main_path, 'config.ini'))

    spark = SparkSession.builder \
        .appName(config['spark']['app_name']) \
        .master(config['spark']['deploy_mode']) \
        .config("spark.driver.cores", config['spark']['driver_cores']) \
        .config("spark.executor.cores", config['spark']['executor_cores']) \
        .config("spark.driver.memory", config['spark']['driver_memory']) \
        .config("spark.executor.memory", config['spark']['executor_memory']) \
        .getOrCreate()

    path_to_data = os.path.join(main_path, config['data']['openfood'])
    preprocessor = Preprocess()

    assembled_data = preprocessor.load_dataset(path_to_data, spark)
    final_data = preprocessor.scale_data(assembled_data)
    kmeans = KMeansModel()
    kmeans.clustering(final_data)

    spark.stop()


if __name__ == '__main__':
    main()
