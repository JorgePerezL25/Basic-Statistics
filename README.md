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

Understanding basics Example:
 For the next code we will explane how Correlation works with the languaje of Scala running in Spark environment.
 


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


         #Correlation
        
        //We import the necesary liberys and packeages to load the program
        import org.apache.spark.ml.linalg.{Matrix, Vectors}
        import org.apache.spark.ml.stat.Correlation
        import org.apache.spark.sql.Row
        
        //Declaring the vectors we will use for this example, the differences beetween declaring a vector as sparse //or a vector       dense is that the dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values. 
        val data = Seq(
          Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
          Vectors.dense(4.0, 5.0, 0.0, 3.0),
          Vectors.dense(6.0, 7.0, 0.0, 8.0),
          Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
        )
        
        // DataFrame data definition
        val df = data.map(Tuple1.apply).toDF("features")
        
        //Differente methods to corralate the data , this one is with pearson
        val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
        println(s"Pearson correlation matrix:\n $coeff1")
        
        //Corralation method Spearman
        val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
        println(s"Spearman correlation matrix:\n $coeff2")

        #Results
        
        data: Seq[org.apache.spark.ml.linalg.Vector] = List((4,[0,3],[1.0,-2.0]), [4.0,5.0,0.0,3.0], [6.0,7.0,0.0,8.0], (4,[0,3],[9.0,1.0]))
        df: org.apache.spark.sql.DataFrame = [features: vector]
        +--------------------+
        |            features|
        +--------------------+
        |(4,[0,3],[1.0,-2.0])|
        |   [4.0,5.0,0.0,3.0]|
        |   [6.0,7.0,0.0,8.0]|
        | (4,[0,3],[9.0,1.0])|
        +--------------------+

        [Stage 2:>(0 + 2) / 2]19/11/11 23:56:29 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
        19/11/11 23:56:29 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
        19/11/11 23:56:30 WARN PearsonCorrelation: Pearson correlation matrix contains NaN values.
        coeff1: org.apache.spark.ml.linalg.Matrix =
        1.0                   0.055641488407465814  NaN  0.4004714203168137
        0.055641488407465814  1.0                   NaN  0.9135958615342522
        NaN                   NaN                   1.0  NaN
        0.4004714203168137    0.9135958615342522    NaN  1.0
        Pearson correlation matrix:
         1.0                   0.055641488407465814  NaN  0.4004714203168137  
        0.055641488407465814  1.0                   NaN  0.9135958615342522  
        NaN                   NaN                   1.0  NaN                 
        0.4004714203168137    0.9135958615342522    NaN  1.0                 
        19/11/11 23:56:32 WARN PearsonCorrelation: Pearson correlation matrix contains NaN values.
        coeff2: org.apache.spark.ml.linalg.Matrix =
        1.0                  0.10540925533894532  NaN  0.40000000000000174
        0.10540925533894532  1.0                  NaN  0.9486832980505141
        NaN                  NaN                  1.0  NaN
        0.40000000000000174  0.9486832980505141   NaN  1.0
        Spearman correlation matrix:
         1.0                  0.10540925533894532  NaN  0.40000000000000174  
        0.10540925533894532  1.0                  NaN  0.9486832980505141   
        NaN                  NaN                  1.0  NaN                  
        0.40000000000000174  0.9486832980505141   NaN  1.0        




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

15211255 - Perez Lomeli Jorge Lorenzo
