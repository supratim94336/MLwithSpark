package SparkDataFrameTutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.functions._


object DataFrameBasics3 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
    val df = spark.read.option("header","true").option("InferSchema","true").csv("../MLwithSpark/Data/CitiGroup2006_2008")
    df.select(year(df("Date"))).show()
    df.select(month(df("Date"))).show()
    val dfYear = df.withColumn("Year", year(df("Date")))
    dfYear.show()
    val dfAvg = dfYear.groupBy("Year").mean("Close")
    dfAvg.show()
    val dfMin = dfYear.groupBy("Year").min("Close")
    dfMin.show()
  }
}
