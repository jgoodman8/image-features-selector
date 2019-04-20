package ifs.unit.jobs

import ifs.jobs.ClassificationPipeline
import ifs.services.{ConfigurationService, DataService}
import ifs.{Constants, TestUtils}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ClassificationPipelineTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val trainFile: String = TestUtils.getTestDataRoute
  val testFile: String = TestUtils.getTestDataRoute
  val modelsPath: String = TestUtils.getModelsRoute
  val metricsPath: String = TestUtils.getMetricsOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
    TestUtils.clearDirectory(modelsPath)
    TestUtils.clearDirectory(metricsPath)
  }

  "trainPipeline" should "classify the dataset using a Logistic Regression model" in {
    val method = Constants.LOGISTIC_REGRESSION
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)
    assert(metrics.count() == ConfigurationService.Model.getMetrics.length)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a Random Forest model" in {
    val method = Constants.RANDOM_FOREST
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)
    assert(metrics.count() == ConfigurationService.Model.getMetrics.length)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a Decision Tree Classifier" in {
    val method = Constants.DECISION_TREE
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)
    assert(metrics.count() == ConfigurationService.Model.getMetrics.length)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a MultiLayer Perceptron" in {
    val method = Constants.MLP
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)
    assert(metrics.count() == ConfigurationService.Model.getMetrics.length)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a Naive Bayes Classifier" in {
    val method = Constants.NAIVE_BAYES
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)
    assert(metrics.count() == ConfigurationService.Model.getMetrics.length)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }
}
