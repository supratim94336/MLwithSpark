package mltutorials
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

object PCA1 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    // global parameters
    Logger.getLogger("org").setLevel(Level.ERROR)

    // create spark session
    val spark = SparkSession.builder
                            .appName("PCA1")
                            .master("local[*]")
                            .getOrCreate()
    // creates an Array, where first part is sparse vector of size 5 and pos 1 it is 1.0 and at pos 3 it is 7.0
    val data = Array(Vectors.sparse(5, Seq((1,1.0),(3,7.0))),
                     Vectors.dense(2.0,0.0,3.0,4.0,5.0),
                     Vectors.dense(1.0,2.0,3.0,4.0,5.0),
                     Vectors.sparse(5, Seq((2,1.0),(4,7.0))))

    // convert this sample dataset into features
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    // creating pca features
    val pca = new PCA().setInputCol("features").setOutputCol("PCAFeatures").setK(3).fit(df)
    // model transform
    val result = pca.transform(df)
    val features = result.select("PCAFeatures")
    result.show()
  }
}
