from pyspark.ml.feature import VectorAssembler, StandardScaler

from db import Database

from logger import Logger

SHOW_LOG = True

class Preprocess:
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    def load_dataset(self, db: Database):
        dataset = db.read_table("OpenFoodFacts")
        #dataset = spark.read.csv(path_to_data, header=True, inferSchema=True)

        vector_assembler = VectorAssembler(
            inputCols=dataset.columns,
            outputCol='features',
            handleInvalid='skip'
        )

        assembled_data = vector_assembler.transform(dataset)
        self.log.info("Assembled data")
        return assembled_data

    def scale_data(self, assembled_data):
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
        scaler_model = scaler.fit(assembled_data)
        data = scaler_model.transform(assembled_data)
        self.log.info("Scaled data")
        return data