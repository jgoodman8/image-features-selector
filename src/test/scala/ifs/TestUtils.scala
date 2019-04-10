package ifs

import java.io.File

import org.apache.spark.sql.SparkSession

import scala.reflect.io.Directory

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

  def getFeaturesOutputRoute: String = "output/output_data"

  def getMetricsOutputRoute: String = "output/metrics"

  def getModelsRoute: String = "output/models"

  def clearDirectory(path: String): Unit = new Directory(new File(path)).deleteRecursively()

  def findFileByWildcard(basePath: String, pattern: String = ""): String = {
    val fileName = new File(basePath).listFiles()
      .filter(file => file.isDirectory)
      .map(file => file.getName)
      .filter(fileName => fileName.contains(pattern))(0)

    basePath + "/" + fileName
  }
}
