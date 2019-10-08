// Import the libraries
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.SparkSession

// Create a SparkSession
val spark = SparkSession.builder.getOrCreate()

// Import the Spark Implicits librarie
import spark.implicits._
import Summarizer._

// DataFrame data definition
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0)
  )

// DataFrame creation
val df = data.toDF("features", "weight")

// Show DataFrame
df.show()

/*
 * Results:
 
  +-------------+------+
  |     features|weight|
  +-------------+------+
  |[2.0,3.0,5.0]|   1.0|
  |[4.0,6.0,7.0]|   2.0|
  +-------------+------+
*/

// Calculate the mean and variance of the dataframe columns using weightCol
val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()

// Show results using weightCol
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

/*
 * Results:

  meanVal: org.apache.spark.ml.linalg.Vector = [3.333333333333333,5.0,6.333333333333333]
  varianceVal: org.apache.spark.ml.linalg.Vector = [2.000000000000001,4.5,2.000000000000001]
  with weight: mean = [3.333333333333333,5.0,6.333333333333333], variance = [2.000000000000001,4.5,2.000000000000001]
*/

// Calculate the mean and variance of the dataframe columns without using weightCol
val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features")).as[(Vector, Vector)].first()

// Show results without using weightCol
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

/*
 * Results:

  meanVal2: org.apache.spark.ml.linalg.Vector = [3.0,4.5,6.0]
  varianceVal2: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]
  without weight: mean = [3.0,4.5,6.0], sum = [2.0,4.5,2.0]
*/

// Stop SparkSession
spark.stop()