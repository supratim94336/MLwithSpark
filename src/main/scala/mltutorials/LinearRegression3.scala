package mltutorials
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegression3 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/Clean-Ecommerce.csv"
    // create spark session
    val spark = SparkSession.builder
      .appName("LinearRegression3")
      .master("local[*]")
      .getOrCreate()
    /**
      * Columns
      * Email, Address, Avatar, Avg Session Length, Time on App, Time on Website, Length of Membership,
      * Yearly Amount Spent
      */
    // second way of reading
    val trainSet = spark.read.option("inferSchema", "true").option("header", "true").format("csv").csv(dataPath)
    trainSet.printSchema()

    // Now for machine learning we need labels and features
    import spark.implicits._
    val trainDf = (trainSet.select(trainSet("Yearly Amount Spent").as("label"),
        $"Avg Session Length"
      , $"Time on App"
      , $"Time on Website"
      , $"Length of Membership"
    ))
    trainDf.na.fill(0.0)
    //    trainDf.show(5)
    val assembler = (new VectorAssembler().setInputCols(Array(
        "Avg Session Length"
      , "Time on App"
      , "Time on Website"
      , "Length of Membership"
    )).setOutputCol("features"))

    val assemblerOut = assembler.transform(trainDf).select($"label", $"features")
    //assemblerOut.show(5)

    val linReg = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    val model = linReg.fit(assemblerOut)
    println(s"Coefficients: ${model.coefficients}; Intercepts: ${model.intercept}")
    val summary = model.summary
    println(s"Total no. of iterations ${summary.totalIterations}")
    println(s"History: ${summary.objectiveHistory.toList}")
    println(s"RMS: ${summary.rootMeanSquaredError}")
    println(s"R2: ${summary.r2}")
    summary.residuals.show()

    spark.stop()
  }
}
