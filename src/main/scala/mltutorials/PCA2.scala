package mltutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors

object PCA2 {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val dataPath = "../MLwithSpark/Data/Cancer_Data"

    val spark = SparkSession.builder
      .appName("Clustering1")
      .master("local[*]")
      .getOrCreate()

    var trainSet = spark.read
                        .option("inferSchema", "true")
                        .option("header", "true")
                        .format("csv")
                        .csv(dataPath)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    trainSet.printSchema()
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    trainSet.describe().show()
    import spark.implicits._
    //trainSet.columns.toList.foreach(println)
    /**
      * root
      * |-- mean radius: integer (nullable = true)
      * |-- mean texture: double (nullable = true)
      * |-- mean perimeter: double (nullable = true)
      * |-- mean area: double (nullable = true)
      * |-- mean smoothness: double (nullable = true)
      * |-- mean compactness: double (nullable = true)
      * |-- mean concavity: double (nullable = true)
      * |-- mean concave points: double (nullable = true)
      * |-- mean symmetry: double (nullable = true)
      * |-- mean fractal dimension: double (nullable = true)
      * |-- radius error: double (nullable = true)
      * |-- texture error: double (nullable = true)
      * |-- perimeter error: double (nullable = true)
      * |-- area error: double (nullable = true)
      * |-- smoothness error: double (nullable = true)
      * |-- compactness error: double (nullable = true)
      * |-- concavity error: double (nullable = true)
      * |-- concave points error: double (nullable = true)
      * |-- symmetry error: double (nullable = true)
      * |-- fractal dimension error: double (nullable = true)
      * |-- worst radius: double (nullable = true)
      * |-- worst texture: double (nullable = true)
      * |-- worst perimeter: double (nullable = true)
      * |-- worst area: double (nullable = true)
      * |-- worst smoothness: double (nullable = true)
      * |-- worst compactness: double (nullable = true)
      * |-- worst concavity: double (nullable = true)
      * |-- worst concave points: double (nullable = true)
      * |-- worst symmetry: double (nullable = true)
      * |-- worst fractal dimension: double (nullable = true)
      */

    val dataImport = (trainSet.select(
                        $"mean radius"
                             ,$"mean texture"
                             ,$"mean perimeter"
                             ,$"mean area"
                             ,$"mean smoothness"
                             ,$"mean compactness"
                             ,$"mean concavity"
                             ,$"mean concave points"
                             ,$"mean symmetry"
                             ,$"mean fractal dimension"
                             ,$"radius error"
                             ,$"texture error"
                             ,$"perimeter error"
                             ,$"area error"
                             ,$"smoothness error"
                             ,$"compactness error"
                             ,$"concavity error"
                             ,$"concave points error"
                             ,$"symmetry error"
                             ,$"fractal dimension error"
                             ,$"worst radius"
                             ,$"worst texture"
                             ,$"worst perimeter"
                             ,$"worst area"
                             ,$"worst smoothness"
                             ,$"worst compactness"
                             ,$"worst concavity"
                             ,$"worst concave points"
                             ,$"worst symmetry"
                             ,$"worst fractal dimension"))

    // drop na
    val dataImportClean = dataImport.na.drop()
    // setting the features
    val assembler = new VectorAssembler()
                        .setInputCols(Array("mean radius",
                                            "mean texture",
                                            "mean perimeter",
                                            "mean area",
                                            "mean smoothness",
                                            "mean compactness",
                                            "mean concavity",
                                            "mean concave points",
                                            "mean symmetry",
                                            "mean fractal dimension",
                                            "radius error",
                                            "texture error",
                                            "perimeter error",
                                            "area error",
                                            "smoothness error",
                                            "compactness error",
                                            "concavity error",
                                            "concave points error",
                                            "symmetry error",
                                            "fractal dimension error",
                                            "worst radius",
                                            "worst texture",
                                            "worst perimeter",
                                            "worst area",
                                            "worst smoothness",
                                            "worst compactness",
                                            "worst concavity",
                                            "worst concave points",
                                            "worst symmetry",
                                            "worst fractal dimension"))
                        .setOutputCol("features")

    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

    val pca = (new PCA()
                  .setInputCol("scaledFeatures")
                  .setOutputCol("pcaFeatures")
                  .setK(4))
    val pipeline = new Pipeline().setStages(Array(assembler, scaler, pca))
    // train the models
    val model  = pipeline.fit(dataImportClean)
    var output = model.transform(dataImportClean)
    // Check out the results
    val result = output.select("pcaFeatures")
    result.show()
    result.head(1).toList.foreach(println)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
  }
}
