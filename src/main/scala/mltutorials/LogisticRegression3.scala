package mltutorials

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.catalyst.expressions.Length

object LogisticRegression3 {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/advertising.csv"
    // create spark session
    val spark = SparkSession.builder
                            .appName("LogisticRegression3")
                            .master("local[*]")
                            .getOrCreate()
    // second way of reading
    var trainSet = spark.read.option("inferSchema", "true").option("header", "true").format("csv").csv(dataPath)
    trainSet.printSchema()
    import spark.implicits._
    /**
      *  Daily Time Spent on Site - Double
      * ,Age - Integer
      * ,Area Income - Double
      * ,Daily Internet Usage - Double
      * ,Ad Topic Line - String
      * ,City - String
      * ,Male - Integer
      * ,Country - String
      * ,Timestamp - not required
      * ,Clicked on Ad - Integer
      */
    trainSet = trainSet.withColumn("Length of Ad", length(col("Ad Topic Line")))
    trainSet = trainSet.withColumn("Hour", hour(col("Timestamp")))
    val dataImport = (trainSet.select(trainSet("Clicked on Ad")
                      .as("label")
                      ,$"Daily Time Spent on Site"
                      ,$"Age"
                      ,$"Area Income"
                      ,$"Daily Internet Usage"
                      ,$"Length of Ad"
                      ,$"City"
                      ,$"Male"
                      ,$"Country"
                      ,$"Hour"))

    // drop na
    val dataImportClean = dataImport.na.drop()

    // encode string columns into numeric columns
    val cityIndexer = new StringIndexer()
                          .setInputCol("City")
                          .setOutputCol("CityIndexed")
                          .setHandleInvalid("keep")
    val countryIndexer = new StringIndexer()
                             .setInputCol("Country")
                             .setOutputCol("CountryIndexed")
                             .setHandleInvalid("keep")
    // encode numeric encoding to one-hot encoding
    val vectorOneHot = new OneHotEncoderEstimator()
                           .setInputCols(Array("CityIndexed","CountryIndexed"))
                           .setOutputCols(Array("CityVec","CountryVec"))
    // setting the features
    val assembler = new VectorAssembler()
                        .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage",
                        "Length of Ad", "CityVec", "Male", "CountryVec", "Hour"))
                        .setOutputCol("features")
    // train and test split
    val Array(train, test) = dataImportClean.randomSplit(Array(0.7, 0.3), seed=12345)

    // logistic regression
    val lr = new LogisticRegression()
    // setting the pipeline
    val pipeline = new Pipeline().setStages(Array(cityIndexer, countryIndexer, vectorOneHot, assembler, lr))
    // train the models
    val model  = pipeline.fit(train)

    // evaluate the test set
    val results = model.transform(test)
    results.show(5)

    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("--------------> Confusion Matrix <----------------")
    println(metrics.confusionMatrix)
    println("--------------> Accuracy <----------------")
    println(metrics.accuracy)
  }
}
