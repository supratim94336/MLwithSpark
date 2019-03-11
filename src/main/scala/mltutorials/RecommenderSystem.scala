package mltutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._

object RecommenderSystem extends java.io.Serializable{
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    // data downloaded from https://grouplens.org/datasets/movielens/20m/
    val dataPath = "C:\\Users\\supratimdas\\Downloads\\ml-20m\\ml-20m\\ratings.csv"
    val movieDescPath = "C:\\Users\\supratimdas\\Downloads\\ml-20m\\ml-20m\\movies.csv"

    val spark = SparkSession.builder
                            .appName("Clustering1")
                            .master("local[*]")
                            .getOrCreate()

    var trainSet = spark.read
                        .option("inferSchema", "true")
                        .option("header", "true")
                        .format("csv")
                        .csv(dataPath)
    var movies = spark.read
                      .option("inferSchema", "true")
                      .option("header", "true")
                      .format("csv")
                      .csv(movieDescPath)
    println("+--------------------+--------------------+--------------------+--------------------+-------------------+")
    trainSet.printSchema()
    movies.printSchema()
    import spark.implicits._
    val moviesMap = movies.drop($"genres")
    val genresMap = movies.drop($"title")
    val moviesMappings = moviesMap.rdd.map(row => row.getInt(0) -> row.getString(1)).collectAsMap()
    val genreMappings = genresMap.rdd.map(row => row.getInt(0) -> row.getString(1).split('|').toList).collectAsMap()
    //moviesMappings.toList.foreach(println)
    //println(moviesMappings(91073))
    //Million Dollar Legs (1932)
    //genreMappings.toList.foreach(println)
    //println(genreMappings(91073))
    //List(Comedy)
    /**
      * If you want to find out the recommendation learning with als, uncomment the below lines
      */
    //Recommendations
    val Array(train, test) = trainSet.randomSplit(Array(0.7,0.3))
    val als = new ALS()
                  .setMaxIter(10)
                  .setRegParam(0.01)
                  .setUserCol("userId")
                  .setItemCol("movieId")
    val model = als.fit(train)
    val predictions = model.transform(test)
    predictions.show(5)
    val error = predictions.select(abs($"rating" - $"prediction"))
    error.na.drop().describe().show()

    /**
      * If you want to find out the top 10 movies, uncomment the below lines
      */
    //Top movies
    val topMovieIDS = trainSet
                      .groupBy("movieId")
                      .count()
                      .orderBy(desc("count"))
                      .cache()
    val top10 = topMovieIDS.take(10)
    for(result <- top10) {
      println(moviesMappings.get(result(0).asInstanceOf[Int]))
    }
  }
}
