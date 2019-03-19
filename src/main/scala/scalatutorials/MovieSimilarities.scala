package scalatutorials
import org.apache.log4j.Level
import org.apache.log4j._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import utilities.UtilFunctions.manOf
import scala.math.sqrt

object MovieSimilarities {
  // data downloaded from https://grouplens.org/datasets/movielens/20m/
  val ratingsPath = "C:\\Users\\supratimdas\\Downloads\\ml-latest-small\\ml-latest-small\\ratings.csv"
  val movieDescPath = "C:\\Users\\supratimdas\\Downloads\\ml-latest-small\\ml-latest-small\\movies.csv"
  // custom types
  type MovieRating = (Int, Double)
  type UserRatingPair = (Int, (MovieRating, MovieRating))
  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]
  // verbose
  def filterDuplicates(userRatings:UserRatingPair):Boolean = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2
    val movie1 = movieRating1._1
    val movie2 = movieRating2._1
    movie1 < movie2
  }
  // (Int, (Double,Double), Int, (Double, Double)) => ((Int, Int),(Double, Double))
  def makePairs(userRatings:UserRatingPair): ((Int, Int),(Double, Double)) = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2
    val movie1 = movieRating1._1
    val rating1 = movieRating1._2
    val movie2 = movieRating2._1
    val rating2 = movieRating2._2
    ((movie1, movie2), (rating1, rating2))
  }

  def computeCosineSimilarity(ratingPairs:RatingPairs): (Double, Int) = {
    var numPairs:Int = 0
    var sum_xx:Double = 0.0
    var sum_yy:Double = 0.0
    var sum_xy:Double = 0.0

    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2

      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingY
      numPairs += 1
    }

    val numerator:Double = sum_xy
    val denominator = sqrt(sum_xx) * sqrt(sum_yy)

    var score:Double = 0.0
    if (denominator != 0) {
      score = numerator / denominator
    }

    (score, numPairs)
  }

  def main(args: Array[String]): Unit = {
    val log = Logger.getLogger("org")
    log.setLevel(Level.ERROR)
    val spark = SparkSession.builder
                            .appName("LinearRegression")
                            .master("local[*]")
                            .getOrCreate()
    // loading movie names
    var movies = spark.read
                      .option("inferSchema", "true")
                      .option("header", "true")
                      .format("csv")
                      .csv(movieDescPath)
    val rowsMovies: RDD[Row] = movies.rdd
    // Each row to -> movie ID => rating
    val movieDict = rowsMovies.map(l => (l(0).asInstanceOf[Int],l(1).asInstanceOf[String])).collectAsMap()
    // println(manOf(movieDict))
    // scala.collection.Map[Int, String]
    // movieDict.foreach(println)
    var ratings = spark.read
                      .option("inferSchema", "true")
                      .option("header", "true")
                      .format("csv")
                      .csv(ratingsPath)
    // println(movieDict.get(70871))
    val rowsRatings: RDD[Row] = ratings.rdd
    // Map ratings: user ID => (movie ID, rating)
    val ratingsRecords = rowsRatings.map(l => (l(0).asInstanceOf[Int], (l(1).asInstanceOf[Int], l(2).asInstanceOf[Double])))
    // ratingsRecords.foreach(println)
    // At this point our RDD consists of: user ID => ((movie ID, rating), (movie ID, rating))
    val joinedRatings = ratingsRecords.join(ratingsRecords)
    // filter what's true
    val uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
    // We now have: (movie1, movie2) => (rating1, rating2)
    val moviePairs = uniqueJoinedRatings.map(makePairs)
    // Now key by (movie1, movie2) pairs to have Interable[rating 1, rating 2] row is by user ID
    val moviePairRatings = moviePairs.groupByKey()
    // compute cosine similarities: understand mapValues
    // val result: RDD[(A, C)] = rdd.map {case (k, v) => (k, f(v))}
    // val result: RDD[(A, C)] = rdd.mapValues(f)
    // here RDD[Int, Iterable[Double, Double]] -> RDD[Int, (Double, Int)]
    val moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()
//    moviePairSimilarities.foreach(println)

    // example
    // calculate similar movies for movie ID = 50
    val scoreThreshold = 0.97
    val coOccurenceThreshold = 50.0
    val movieID = 50
    val filteredResults = moviePairSimilarities.filter(x => {
      val pair = x._1
      val sim = x._2
      (pair._1 == movieID || pair._2 == movieID) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
    }
    )
    // Sort by quality score.
    val results = filteredResults.map(x => (x._2, x._1)).sortByKey(false).take(10)
    println("\nTop 10 similar movies for " + movieDict.get(movieID))
    for (result <- results) {
      val sim = result._1
      val pair = result._2
      // Display the similarity result that isn't the movie we're looking at
      var similarMovieID = pair._1
      if (similarMovieID == movieID) {
        similarMovieID = pair._2
      }
      println(movieDict.get(similarMovieID) + "\tscore: " + sim._1 + "\tstrength: " + sim._2)
    }
  }
}
