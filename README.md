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

# Hypothesis testing 
Hypothesis testing is a powerful tool in statistics to determine if a result is statistically significant, if this result occurs by chance or not.

All hypotheses are tested by a four step process:
1. The first step is for the analyst to establish the two hypotheses so that only one can be correct.
2. The next step is to formulate an analysis plan, which describes how the data will be evaluated.
3. The third step is to carry out the plan and physically analyze the sample data.
4. The fourth and final step is to analyze the results and accept or reject the null hypothesis.

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

        # Hypothesis testing
        
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

14212347 - Zavala Zuñiga Lineth


