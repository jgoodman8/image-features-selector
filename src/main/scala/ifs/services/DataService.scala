package ifs.services

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object DataService {

  def load(sparkSession: SparkSession, fileRoute: String): DataFrame = {

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileRoute)
  }

  def save(data: DataFrame, fileDir: String, label: String, features: String): Unit = {
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

}
