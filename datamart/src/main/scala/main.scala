import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.sql.functions._
import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import scala.io.StdIn

object DataMart {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DataMart")
      .master("local[*]")
      .getOrCreate()

    val filePath = "data/openfoodfacts.csv"
    val df = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)

    println(s"Preprocessing completed.")
    val assemble = assembleVector(df)
    val scaled = scaleAssembledDataset(assemble)
    scaled.write.mode(SaveMode.Overwrite).parquet("/shared/scaled_data.parquet")
    while (true) {
      Thread.sleep(1000) // Пауза на 1 секунду
    }

  }

  def assembleVector(df: DataFrame): DataFrame = {
    println("Assembling vector from DataFrame columns")
    val inputCols = Array(
      "energy-kcal_100g", "energy_100g", "fat_100g", "saturated-fat_100g",
      "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g",
      "sodium_100g", "fruits-vegetables-nuts-estimate-from-ingredients_100g"
    )
    val outputCol = "features"

    val vectorAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol(outputCol)
    val assembledDf = vectorAssembler.transform(df)
    println("Assembled vector schema: " + assembledDf.schema)

    assembledDf
  }

  def scaleAssembledDataset(df: DataFrame): DataFrame = {
    println("Scaling assembled dataset")
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled")
    val scalerModel = scaler.fit(df)
    val scaledDf = scalerModel.transform(df)
    println("Scaled dataset schema: " + scaledDf.schema)

    scaledDf
  }

  def writeData(df: DataFrame): Unit = {
    // JDBC URL for MySQL database
    val url = "jdbc:mysql://mysql:3306/lab6_bd"
    // Drop the vector columns to avoid the complex type issue
    val dfToWrite = df.drop("features").drop("scaled")

    // Write preprocessed data to MySQL database
    dfToWrite.write.format("jdbc")
      .option("url", url)
      .option("driver", "com.mysql.cj.jdbc.Driver")
      .option("dbtable", "openfoodfacts")
      .option("user", "root")
      .option("password", "0000")
      .mode("overwrite")
      .save()
  }

  def readData(spark: SparkSession): DataFrame = {
    // JDBC URL for MySQL database
    val url = "jdbc:mysql://mysql:3306/lab6_bd"

    // Read data from MySQL table "openfoodfacts"
    val mysqlData = spark.read.format("jdbc")
      .option("url", url)
      .option("driver", "com.mysql.cj.jdbc.Driver")
      .option("dbtable", "openfoodfacts")
      .option("user", "root")
      .option("password", "0000")
      .load()
    mysqlData
  }


}
