package sparkdataframetutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.functions._


object DataFrameExcercise extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    // create spark session
    val spark = SparkSession.builder
                            .appName("SparkSQL")
                            .master("local[*]")
                            .getOrCreate()
    // Read dataset
    val df = spark.read.option("header","true").option("InferSchema","true").csv("../MLwithSpark/Data/Netflix_2011_2016.csv").as("dfMain")
    // Infer Data type
    df.printSchema()
    // describe schema
    df.describe()
    // print top 5 rows
    // all columns
    println(df.columns.toSeq) // or import scala.runtime.ScalaRunTime._ --> stringOf(df.columns)
    // Column names and types
    println(df.dtypes.toSeq)
    // New column
    df.withColumn("HV", col("High")/col("Volume")).show()
    // Peak of High Price
    df.orderBy(desc("High"))
      .select("Date","High")
      .show(1)
    // Volume mean, min and max
    df.select(mean(col("Volume")), max(col("Volume")), min(col("Volume"))).show()
    import spark.implicits._
    // filter by value (scalar)
    println(df.filter($"Low"<600).count())
    println(df.filter($"High">500).count() * 1.0 * 100 / df.count())
    df.select(corr($"High",$"volume"))
      .show()
    df.withColumn("Year", year($"Date"))
      .select($"Year", $"High")
      .groupBy($"Year")
      .max()
      .select($"Year",$"max(High)")
      .orderBy($"Year")
      .show()
    df.withColumn("Month", month($"Date"))
      .select($"Month", $"Close")
      .groupBy($"Month")
      .avg()
      .select($"Month",$"avg(Close)")
      .orderBy($"Month")
      .show()

  }
}
