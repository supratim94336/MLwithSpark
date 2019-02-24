package mltutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.regression.LinearRegression


object LinearRegression1 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/sample_linear_regression_data.txt"
    // create spark session
    val spark = SparkSession.builder
                            .appName("LinearRegression")
                            .master("local[*]")
                            .getOrCreate()
    // first way of reading
    val trainSet = spark.read.format("libsvm").load(dataPath)
    // second way of reading
    trainSet.printSchema()
    /***
      * ElasticNetParam:
      * Set the ElasticNet mixing parameter. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1
      * penalty. For alpha in (0,1), the penalty is a combination of L1 and L2. Default is 0.0 which is an L2 penalty.
      */
    // RegParam = ; ElasticNetParam =
    val linReg =  new LinearRegression()
                      .setMaxIter(100)
                      .setRegParam(0.3)
                      .setElasticNetParam(0.8)

    val model = linReg.fit(trainSet)
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

