package ifs

import java.io.File

import ifs.Constants.Classifiers
import ifs.services.{ConfigurationService, DataService}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalactic.TolerantNumerics

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

  def findFileByPattern(basePath: String, pattern: String = ""): String = {
    val fileName = new File(basePath).listFiles()
      .filter(file => file.isDirectory)
      .map(file => file.getName)
      .filter(fileName => fileName.contains(pattern))(0)

    basePath + "/" + fileName
  }

  def checkMetricsFile(filePattern: String, method: String, metricsPath: String, sparkSession: SparkSession): Unit = {
    implicit val custom = TolerantNumerics.tolerantDoubleEquality(0.000001)

    val metricsFile: String = TestUtils.findFileByPattern(metricsPath, filePattern)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)

    if (method == Classifiers.NAIVE_BAYES || method == Classifiers.DECISION_TREE ||
      method == Classifiers.RANDOM_FOREST || method == Classifiers.MLP) {

      val metricsData = metrics.collect()
      assert(metricsData.length == ConfigurationService.Model.getMetrics.length + 1)
      assert(metricsData(0).getDouble(1) == metricsData(1).getDouble(1))

    } else {
      assert(metrics.count() == ConfigurationService.Model.getMetrics.length)
    }
  }

  def checkModelPath(modelsPath: String): Unit = {
    val modelFile: String = TestUtils.findFileByPattern(modelsPath)
    assert(modelFile.nonEmpty)
  }
}
