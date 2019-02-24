package sparkdataframetutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.functions._


object DataFrameBasics2 extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder
                            .appName("DataFrameBasics2")
                            .master("local[*]")
                            .getOrCreate()
    val df = spark.read.option("header","true").option("InferSchema","true").csv("../MLwithSpark/Data/Sales.csv")
    // groupBy - mean/min/max/avg/count
    df.groupBy("Company").avg("Sales").show()
    df.groupBy("Company").count().show()

    // aggregate
    df.select(countDistinct("Sales")).show()
    df.select(sumDistinct("Sales")).show()
    df.select(variance("Sales")).show()
    df.select(stddev("Sales")).show()
    df.select(collect_set("Sales")).show()

    // orderBy
    df.orderBy(desc("Sales")).show()

    // deal with nulls
    // drop all rows that contains na
    df.na.drop()
    // drop all rows containing 2 na values
    df.na.drop(2)
    // fill all integer columns with 0 for na
    df.na.fill(0)
    // fill for a specific column if it's of type String
    df.na.fill("New name", Array("Name"))

    // rename
    val dfMain = df.as("dfMain")

    // experiment
    val maxOfLot = df.groupBy("Company")
                     .max("Sales")
                     .select(col("Company"), col("max(Sales)").alias("SalesMax"))
                     .as("maxOfLot")
    maxOfLot.show()
    val joinedDf = dfMain.join(maxOfLot, dfMain("Company") === maxOfLot("Company"), "inner")
                         .select("dfMain.Company","dfMain.Person", "dfMain.Sales", "maxOfLot.SalesMax")
                         .as("maxOfLot")
    joinedDf.show()

    val parentChildDf = joinedDf.withColumn("Relation", when(col("Sales") === col("SalesMax"), "Parent").otherwise("Child"))
    parentChildDf.show()
  }
}
