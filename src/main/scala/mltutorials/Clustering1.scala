package mltutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.clustering.KMeans

object Clustering1 {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)
    // local parameters
    val dataPath = "../MLwithSpark/Data/sample_kmeans_data.txt"
    // create spark session
    val spark = SparkSession.builder
      .appName("Clustering1")
      .master("local[*]")
      .getOrCreate()
    // first way of reading
    val trainSet = spark.read.format("libsvm").load(dataPath)
    // second way of reading
    trainSet.printSchema()
    // create kmeans instance
    val kmeans = new KMeans().setK(2).setSeed(1L)
    // train kmeans
    val model = kmeans.fit(trainSet)
    val computeCost = model.computeCost(trainSet)
    println(s"Within set sum of squares $computeCost")

    model.clusterCenters.foreach(println)
  }
}
