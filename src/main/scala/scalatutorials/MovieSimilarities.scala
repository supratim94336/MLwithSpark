package scalatutorials

import scala.io.Source

object MovieSimilarities {
  // user defined function to load movie names from csv file
  def loadMovieNames(path: String) : Map[Int, String] = {
    var movieNames:Map[Int, String] = Map()
    val lines = Source.fromFile(path).getLines()
    for(line <- lines) {
      var fields = line.split(',')
      if(fields.length>1){
        try {
          movieNames += (fields(0).toInt -> fields(1))
        } catch {
          case e: NumberFormatException => println("Not a number")
        }
      }
    }
    movieNames
  }

  // custom type
  type MovieRating = (Int, Double)
  type UserRatingPair = (Int, (MovieRating, MovieRating))

  // make pairs of ratings for same user but different movies
  def makePairs(userRatings:UserRatingPair) = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2

    val movie1 = movieRating1._1
    val rating1 = movieRating1._2
    val movie2 = movieRating2._1
    val rating2 = movieRating2._2

    ((movie1, movie2), (rating1, rating2))
  }

  def main(args: Array[String]): Unit = {
    val nameDict = loadMovieNames("C:\\Users\\supratimdas\\Downloads\\ml-20m\\ml-20m\\movies.csv")
    nameDict.foreach(println)
  }
}
