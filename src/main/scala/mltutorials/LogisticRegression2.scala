package mltutorials

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object LogisticRegression2 {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/titanic.csv"
    // create spark session
    val spark = SparkSession.builder
                            .appName("LogisticRegression2")
                            .master("local[*]")
                            .getOrCreate()
    // second way of reading
    val trainSet = spark.read.option("inferSchema", "true").option("header", "true").format("csv").csv(dataPath)
    trainSet.printSchema()

    import spark.implicits._

    val dataImport = (trainSet.select(trainSet("survived").as("label"), $"PClass", $"Sex", $"Age", $"Sibsp",
                                      $"Parch", $"Fare", $"Embarked"))
    val dataImportClean = dataImport.na.drop()
    //dataImportClean.show()
    // encode string columns into numeric columns
    val genderIndexer = new StringIndexer()
                            .setInputCol("Sex")
                            .setOutputCol("SexIndexed")
    val embarkIndexer = new StringIndexer()
                            .setInputCol("Embarked")
                            .setOutputCol("EmbarkIndexed")
    // encode numeric encoding to one-hot encoding
    val vectorOneHot = new OneHotEncoderEstimator()
                           .setInputCols(Array("SexIndexed","EmbarkIndexed"))
                           .setOutputCols(Array("SexVec","EmbarkVec"))
    // setting the features
    val assembler = new VectorAssembler()
                        .setInputCols(Array("PClass", "SexVec", "Age", "Sibsp", "Parch", "Fare", "EmbarkVec"))
                        .setOutputCol("features")
    // train and test split
    val Array(train, test) = dataImportClean.randomSplit(Array(0.7,0.3), seed=12345)

    // logistic regression
    val lr = new LogisticRegression()
    // setting the pipeline
    val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, vectorOneHot, assembler, lr))
    // train the models
    val model  = pipeline.fit(train)

    // evaluate the test set
    val results = model.transform(test)
    results.show(5)

    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("--------------> Confusion Matrix <----------------")
    println(metrics.confusionMatrix)
  }
}
