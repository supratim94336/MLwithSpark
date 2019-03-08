package mltutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, VectorAssembler}

object Clustering2 {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/Wholesale customers data.csv"
    // create spark session
    val spark = SparkSession.builder
                            .appName("Clustering2")
                            .master("local[*]")
                            .getOrCreate()
    // first way of reading
    var trainSet = spark.read
                        .option("inferSchema", "true")
                        .option("header", "true")
                        .format("csv")
                        .csv(dataPath)

    // second way of reading
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    trainSet.printSchema()
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    import spark.implicits._
    /**
      *  Channel
      * ,Region
      * ,Fresh
      * ,Milk
      * ,Grocery
      * ,Frozen
      * ,Detergents_Paper
      * ,Delicassen
      */
    val dataImport = (trainSet.select(
                         $"Channel"
                              ,$"Region"
                              ,$"Fresh"
                              ,$"Milk"
                              ,$"Grocery"
                              ,$"Frozen"
                              ,$"Detergents_Paper"
                              ,$"Delicassen"))

    // drop na
    val dataImportClean = dataImport.na.drop()
    // encode numeric encoding to one-hot encoding
    val vectorOneHot = new OneHotEncoderEstimator()
                           .setInputCols(Array("Channel","Region"))
                           .setOutputCols(Array("ChannelVec","RegionVec"))
    // setting the features
    val assembler = new VectorAssembler()
                        .setInputCols(Array("ChannelVec", "RegionVec", "Fresh", "Milk", "Grocery",
                          "Frozen", "Delicassen", "Detergents_Paper"))
                        .setOutputCol("features")
    // create kmeans instance
    val kmeans = new KMeans().setK(5).setSeed(1L)
    // train kmeans
    // setting the pipeline
    val pipeline = new Pipeline().setStages(Array(vectorOneHot, assembler, kmeans))
    // train the models
    val model  = pipeline.fit(dataImportClean)
    var output = model.transform(dataImportClean)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    output.show(5)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    val centers = model.stages(2).asInstanceOf[KMeansModel].clusterCenters
    centers.foreach(println)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    output.drop("features")
    val wwse = model.stages(2).asInstanceOf[KMeansModel].computeCost(output)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    println(s"Computation cost is is $wwse")
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
  }
}
