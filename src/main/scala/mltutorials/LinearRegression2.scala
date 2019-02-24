package mltutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
// ML specific
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


object LinearRegression2 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/USA_Housing.csv"
    // create spark session
    val spark = SparkSession.builder
                            .appName("LinearRegression2")
                            .master("local[*]")
                            .getOrCreate()

    // second way of reading
    val trainSet = spark.read.option("inferSchema","true").option("header","true").format("csv").csv(dataPath)

//    val colNames = trainSet.columns
//    val firstRow = trainSet.head(1)(0)
//    println()
//    // iterate
//    for(ind <- Range(1, colNames.length)) {
//      println(colNames(ind))
//      println(firstRow(ind))
//    }
    // Now for machine learning we need labels and features
    import spark.implicits._
    val trainDf = (trainSet.select(trainSet("Price").as("label"),
                                   $"Avg Area Income"
                                  ,$"Avg Area House Age"
                                  ,$"Avg Area Number of Rooms"
                                  ,$"Avg Area Number of Bedrooms"
                                  ,$"Area Population"))
//    trainDf.show(5)
    val assembler = (new VectorAssembler().setInputCols(Array("Avg Area Income"
                                                              ,"Avg Area House Age"
                                                              ,"Avg Area Number of Rooms"
                                                              ,"Avg Area Number of Bedrooms"
                                                              ,"Area Population")).setOutputCol("features"))
    val assemblerOut = assembler.transform(trainDf).select($"label", $"features")
    //assemblerOut.show(5)

    val linReg =  new LinearRegression()
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

