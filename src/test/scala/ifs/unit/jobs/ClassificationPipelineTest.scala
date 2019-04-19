package ifs.unit.jobs

import ifs.jobs.ClassificationPipeline
import ifs.services.DataService
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

  "trainPipeline" should "classify the dataset using a logistic regression model" in {
    val method = Constants.LOGISTIC_REGRESSION
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(selectedTest.columns.length == 1)
    assert(selectedTest.count() == 1)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a random forest model" in {
    val method = Constants.RANDOM_FOREST
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 1)
    assert(metrics.count() == 1)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a decision tree" in {
    val method = Constants.DECISION_TREE
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 1)
    assert(metrics.count() == 1)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "classify the dataset using a multi layer perceptron" in {
    val method = Constants.MLP
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 1)
    assert(metrics.count() == 1)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }
}
