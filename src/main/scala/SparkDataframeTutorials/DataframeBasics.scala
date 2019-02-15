package SparkDataframeTutorials
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

object DataframeBasics extends java.io.Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder
                            .appName("SparkSQL")
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
    val df2 = df.withColumn("HighPlusLow", df("High")+df("Low"))
    df2.select("HighPlusLow", "High", "Low").show(5)
    // check schema
    //df2.printSchema()
    // alias
    df2.select(df2("HighPlusLow").as("HPL")).show(5)
    // new changes
  }
}
