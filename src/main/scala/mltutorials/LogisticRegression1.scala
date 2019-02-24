package mltutorials

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object LogisticRegression1 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/sample_libsvm_data.txt"
    // create spark session
    val spark = SparkSession.builder
                            .appName("LogisticRegression1")
                            .master("local[*]")
                            .getOrCreate()
    // second way of reading
    val trainSet = spark.read.format("libsvm").load(dataPath)
    trainSet.printSchema()

    val logReg = new LogisticRegression()
                     .setMaxIter(100)
                     .setRegParam(0.3)
                     .setElasticNetParam(0.8)
    val model = logReg.fit(trainSet)
    println(s"Coefficients: ${model.coefficients}; Intercepts: ${model.intercept}")
    val summary = model.summary
    println(s"Total no. of iterations ${summary.totalIterations}")
    println(s"History: ${summary.objectiveHistory.toList}")

    spark.stop()
  }
}
