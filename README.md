# Basic-Statistics

# Table of content 

    # Correlation
    # Hypothesis testing
    # Summarizer
    # Code & Results
    # Collaborators

# Correlation
Proportionality and the linear relationship that exists between different variables. If the values of one variable are systematically modified with respect to the values of another, it is said that both variables are correlated.

Understanding Correlation
Let's Suppose we have an "R" variable and an "S" variable. As R values increase, S values increase also. Similarly, when S values increase, R values are increased as well. Therefore there is a correlation between the R and S variables.

# Summarizer
It is basically a tool that provides a series of statistics for a given DataFrame.

The available metrics are:
- count: The account of all the vectors seen
- max: The maximum for each coefficient
- mean: A vector that contains the mean in terms of coefficients
- min: The minimum for each coefficient
- normL1: The L1 standard of each coefficient (sum of absolute values)
- normL2: The Euclidean norm for each coefficient
- numNonzeros: A vector with the number of non-zeros for each coefficient
- variance: A vector containing the variance of the wise coefficient

# Code & Results 

    # Summarizer
    
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
    val data = Seq( (Vectors.dense(2.0, 3.0, 5.0), 1.0),
                    (Vectors.dense(4.0, 6.0, 7.0), 2.0) )

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

# Collaborators
15510506 - Cabrera Reyes Ivan


