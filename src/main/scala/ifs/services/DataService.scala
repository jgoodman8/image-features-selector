package ifs.services

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.functions.{col, max, udf}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object DataService {

  def load(sparkSession: SparkSession, fileRoute: String): DataFrame = {

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("maxColumns", ConfigurationService.Session.getMaxCSVLength)
      .option("inferSchema", "true")
      .load(fileRoute)
  }

  def save(data: DataFrame, fileDir: String): Unit = {
    data
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(fileDir + System.currentTimeMillis.toString)
  }

  def extractDenseRows(data: DataFrame, features: String, labels: String): DataFrame = {

    val columnsSize = data.first.getAs[Vector](features).size + 1

    val vecToSeq = udf((v: Vector, label: Double) => v.toArray :+ label)
    val columns = (0 until columnsSize).map(i => col("_tmp").getItem(i).alias(s"f$i"))

    data
      .select(vecToSeq(col(features), col(labels)).alias("_tmp"))
      .select(columns: _*)
  }

  def getNumberOfFeatures(data: DataFrame, features: String): Int = {
    val firstRow = data.first()
    if (firstRow.getAs[Any](features).isInstanceOf[DenseVector]) {
      firstRow.getAs[DenseVector](features).values.length
    } else {
      firstRow.getAs[SparseVector](features).values.length
    }
  }

  def getNumberOfLabels(data: DataFrame, label: String): Int = {
    data.agg(max(label)).first().getAs[Double](0).toInt + 1
  }
}
