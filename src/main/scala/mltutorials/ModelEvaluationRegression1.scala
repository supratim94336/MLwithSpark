package mltutorials

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

object ModelEvaluationRegression1 {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters

    val dataPath = "../MLwithSpark/Data/sample_linear_regression_data.txt"

    // create spark session
    val spark = SparkSession.builder
                            .appName("ModelEvaluationRegression1")
                            .master("local[*]")
                            .getOrCreate()

    // first way of reading
    val trainSet = spark.read.format("libsvm").load(dataPath)

    // second way of reading
    trainSet.printSchema()

    // train test split
    val Array(train, test) = trainSet.randomSplit(Array(0.9,0.1), seed=12345)

    val lr = new LinearRegression()

    // set the parameter grid
    val paramGrid = new ParamGridBuilder()
                        .addGrid(lr.regParam, Array(0.1, 0.01))
                        .addGrid(lr.fitIntercept)
                        .addGrid(lr.elasticNetParam, Array(0, 0.5, 1.0))
                        .build()

    // set validation parameter
    val trainValidationSplit = new TrainValidationSplit()
                                   .setEstimator(lr)
                                   .setEvaluator(new RegressionEvaluator())
                                   .setEstimatorParamMaps(paramGrid)
                                   .setTrainRatio(0.8)

    val model = trainValidationSplit.fit(train)
    model.transform(test).select("features", "label", "prediction").show()
  }
}
