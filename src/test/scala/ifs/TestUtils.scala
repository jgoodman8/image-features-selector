package ifs

import java.io.File

import ifs.Constants.Classifiers
import ifs.services.ConfigurationService
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalactic.TolerantNumerics

import scala.reflect.io.Directory

object TestUtils {

  def getTestSession: SparkSession = {
    SparkSession.builder()
      .appName("Test Session")
      .master("local[4]")
      .config("parquet.compression", "snappy")
      .config("compression", "snappy")
      .config("spark.sql.parquet.compression.codec", "snappy")
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
    val folderName = new File(basePath).listFiles()
      .filter(file => file.isDirectory)
      .map(file => file.getName)
      .filter(fileName => fileName.contains(pattern))(0)

    val baseFolder = basePath + "/" + folderName
    val subFolderNames = new File(baseFolder).listFiles()
      .filter(file => file.isDirectory)
      .map(file => file.getName).toList

    if (subFolderNames.nonEmpty) {
      return baseFolder + "/" + subFolderNames.head
    }

    baseFolder
  }

  def checkMetricsFile(filePattern: String, method: String, metricsPath: String, sparkSession: SparkSession): Unit = {
    implicit val custom = TolerantNumerics.tolerantDoubleEquality(0.0001)

    val metrics: DataFrame = this.getMetricsData(sparkSession, filePattern, metricsPath)

    assert(metrics.columns.length == 2)

    if (method == Classifiers.NAIVE_BAYES || method == Classifiers.DECISION_TREE ||
      method == Classifiers.RANDOM_FOREST || method == Classifiers.MLP) {

      val metricsData = metrics.collect()
      assert(metricsData.length == ConfigurationService.Model.getMetrics.length + 1)
      assert(custom.areEqual(metricsData(0).getDouble(1), metricsData(1).getDouble(1)))

    } else {
      assert(metrics.count() == ConfigurationService.Model.getMetrics.length)
    }
  }

  def checkModelPath(modelsPath: String): Unit = {
    val modelFile: String = TestUtils.findFileByPattern(modelsPath)
    assert(modelFile.nonEmpty)
  }

  private def getMetricsData(sparkSession: SparkSession, filePattern: String, metricsPath: String): DataFrame = {
    val metricsSchema: StructType = StructType(List(StructField("metric", StringType), StructField("value", DoubleType)))
    val metricsFile: String = TestUtils.findFileByPattern(metricsPath, filePattern)
    sparkSession.read.format("csv")
      .option("header", "true")
      .schema(metricsSchema)
      .load(metricsFile)
  }
}
