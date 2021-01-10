package ifs.jobs

import ifs.services.ConfigurationService
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object GenerateDataSample extends App {
  def run(session: SparkSession,
          src: String,
          dst: String,
          howManyTake: Integer): Unit = {

    val data: DataFrame = session.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("maxColumns", ConfigurationService.Session.getMaxCSVLength)
      .option("inferSchema", "true")
      .load(src)
    data.cache()

    val count = data.count()
    val sampledData = data.sample(howManyTake.toFloat / count.toFloat)

    data.unpersist(false)
    sampledData.cache()

    sampledData.write.mode(SaveMode.Overwrite).csv(dst)
  }

  val Array(src: String, dst: String, howManyTake: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName("GenerateDataSample").getOrCreate()

  this.run(sparkSession, src, dst, Integer.parseInt(howManyTake))

  sparkSession.close()
}