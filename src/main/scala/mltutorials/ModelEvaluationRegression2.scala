package mltutorials

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

object ModelEvaluationRegression2 {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/USA_Housing.csv"
    // create spark session
    val spark = SparkSession.builder
      .appName("ModelEvaluationRegression2")
      .master("local[*]")
      .getOrCreate()
    // first way of reading
    var trainSet = spark.read.option("inferSchema", "true").option("header", "true").format("csv").csv(dataPath)
    // second way of reading
//    trainSet.printSchema()
    /**
      * root
      * |-- Avg Area Income: double (nullable = true)
      * |-- Avg Area House Age: double (nullable = true)
      * |-- Avg Area Number of Rooms: double (nullable = true)
      * |-- Avg Area Number of Bedrooms: double (nullable = true)
      * |-- Area Population: double (nullable = true)
      * |-- Price: double (nullable = true)
      */
    import spark.implicits._
    val dataImport = (trainSet.select(trainSet("Price")
                      .as("label")
                      ,$"Avg Area Income"
                      ,$"Avg Area House Age"
                      ,$"Avg Area Number of Rooms"
                      ,$"Avg Area Number of Bedrooms"
                      ,$"Area Population"
                      ))

    // drop na
    val dataImportClean = dataImport.na.drop()

    // setting the features
    val assembler = new VectorAssembler()
                        .setInputCols(Array(
                          "Avg Area Income"
                          ,"Avg Area House Age"
                          ,"Avg Area Number of Rooms"
                          ,"Avg Area Number of Bedrooms"
                          ,"Area Population"))
                        .setOutputCol("features")

    // assembler output
    val assemblerOutput = assembler.transform(dataImportClean).select($"label", $"features")

    // train test split
    val Array(train, test) = assemblerOutput.select($"label", $"features").randomSplit(Array(0.7,0.3), seed=12345)
    val lr = new LinearRegression()

    // set the parameter grid
    val paramGrid = new ParamGridBuilder()
                        .addGrid(lr.regParam, Array(1000, 0.1))
                        .build()

    // set validation parameter
    val trainValidationSplit = new TrainValidationSplit()
                                   .setEstimator(lr)
                                   .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                                   .setEstimatorParamMaps(paramGrid)
                                   .setTrainRatio(0.8)

    val model = trainValidationSplit.fit(train)

    model.transform(test).select("features", "label", "prediction").show()
    println(model.validationMetrics.toList)
  }
}
