package ifs

import org.apache.spark.sql.SparkSession

object TestUtils {

  def getTestSession: SparkSession = {
    SparkSession.builder()
      .appName("Test Session")
      .master("local[4]")
      .config("spark.driver.memory", "6g")
      .config("spark.executor.memory", "6g")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.shuffle.spill", "false")
      .getOrCreate()
  }

  def getTestDataRoute: String = "data/test_data.csv"

  def getFeaturesOutputRoute: String = "output/output_data.csv"

  def getMetricsOutputRoute: String = "output/metrics.csv"

  def getModelsRoute: String = "output/models"
}
