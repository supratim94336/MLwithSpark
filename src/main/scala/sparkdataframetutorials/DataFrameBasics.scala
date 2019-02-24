package sparkdataframetutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.functions._


object DataFrameBasics extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder
                            .appName("DataFrameBasics")
                            .master("local[*]")
                            .getOrCreate()
    val df = spark.read.option("header","true").option("InferSchema","true").csv("../MLwithSpark/Data/CitiGroup2006_2008")
    // print the dataframe
    //df.show(5)
    // show summary of the dataframe
    //df.describe().show()
    // show only one column
    //df.select("Date", "Volume").show(5)
    // create a new column based on existing column
    val df2 = df.withColumn("HighPlusLow", df("High") + df("Low"))
    df2.select("HighPlusLow", "High", "Low").show(5)
    // check schema
    //df2.printSchema()
    // alias
    df2.select(df2("HighPlusLow").as("HPL")).show(5)

    // new changes
    import spark.implicits._
    // Scala syntax (Scala Dataframe) spark.implicits._
    df.filter($"Close" < 480 && $"High" < 480).show(5)
    // SQL syntax (Spark SQL Dataframe) org.apache.spark.sql.Dataframe
    df.filter("Close < 480 AND High < 480").show(5)
    // understanding ===
    df.filter($"High" === 480).show(5)
    // SQL syntax for the same
    df.filter("High = 484.40").show()
    // calculating correlation
    df.select(corr("High","Low")).show()

  }
}
