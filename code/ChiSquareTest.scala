// Import the libraries
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.SparkSession

// Create a SparkSession
val spark = SparkSession.builder.getOrCreate()

// Import the Spark Implicits librarie
import spark.implicits._

// DataFrame data definition
val data = Seq((0.0, Vectors.dense(0.5, 10.0)),
               (0.0, Vectors.dense(1.5, 20.0)),
               (1.0, Vectors.dense(1.5, 30.0)),
               (0.0, Vectors.dense(3.5, 30.0)),
               (0.0, Vectors.dense(3.5, 40.0)),
               (1.0, Vectors.dense(3.5, 40.0)))

// DataFrame creation
val df = data.toDF("label", "features")

// Show DataFrame
df.show()

/*
 * Results:

  +-----+----------+
  |label|  features|
  +-----+----------+
  |  0.0|[0.5,10.0]|
  |  0.0|[1.5,20.0]|
  |  1.0|[1.5,30.0]|
  |  0.0|[3.5,30.0]|
  |  0.0|[3.5,40.0]|
  |  1.0|[3.5,40.0]|
  +-----+----------+
*/

// Invoque test function from ChiSquareTest class
val chi = ChiSquareTest.test(df, "features", "label").head

// Show results
println(s"pValues = ${chi.getAs[Vector](0)}")
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
println(s"statistics ${chi.getAs[Vector](2)}")
/*
 * Results:

  chi: org.apache.spark.sql.Row = [[0.6872892787909721,0.6822703303362126],WrappedArray(2, 3),[0.75,1.5]]
  pValues = [0.6872892787909721,0.6822703303362126]
  degreesOfFreedom [2,3]
  statistics [0.75,1.5]
*/

// Stop SparkSession
spark.stop()