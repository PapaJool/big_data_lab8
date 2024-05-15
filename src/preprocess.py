from pyspark.ml.feature import VectorAssembler, StandardScaler

import src.db


class Preprocess:
    def load_dataset(self, db: src.db.Database):
        dataset = db.read_table("OpenFoodFacts")
        #dataset = spark.read.csv(path_to_data, header=True, inferSchema=True)

        vector_assembler = VectorAssembler(
            inputCols=dataset.columns,
            outputCol='features',
            handleInvalid='skip'
        )

        assembled_data = vector_assembler.transform(dataset)

        return assembled_data

    def scale_data(self, assembled_data):
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
        scaler_model = scaler.fit(assembled_data)
        data = scaler_model.transform(assembled_data)

        return data