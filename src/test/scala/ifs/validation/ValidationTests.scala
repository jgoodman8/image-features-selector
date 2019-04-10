package ifs.validation

import ifs.jobs.{ClassificationPipeline, FeatureSelectionPipeline}
import ifs.services.DataService
import ifs.{Constants, TestUtils}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ValidationTests extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val trainFile: String = TestUtils.getTestDataRoute
  val testFile: String = TestUtils.getTestDataRoute
  val featuresPath: String = TestUtils.getFeaturesOutputRoute
  val modelsPath: String = TestUtils.getModelsRoute
  val metricsPath: String = TestUtils.getMetricsOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
    TestUtils.clearDirectory(featuresPath)
    TestUtils.clearDirectory(modelsPath)
    TestUtils.clearDirectory(metricsPath)
  }

  it should "perform the full pipeline (ChiSQ + Logistic Regression)" in {
    val numFeatures = 3
    val featureSelectionMethod = Constants.CHI_SQ
    val classificationMethod = Constants.LOGISTIC_REGRESSION

    FeatureSelectionPipeline.run(sparkSession, trainFile, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByWildcard(featuresPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.getDataFromFile(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByWildcard(featuresPath, pattern = "test")
    val selectedTest: DataFrame = DataService.getDataFromFile(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, outputTest, metricsPath, modelsPath, classificationMethod)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.getDataFromFile(sparkSession, metricsFile)
    assert(metrics.columns.length == 1)
    assert(metrics.count() == 1)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }

  it should "perform the full pipeline (mRMR Selection + Random Forest)" in {
    val numFeatures = 3
    val featureSelectionMethod = Constants.MRMR
    val classificationMethod = Constants.RANDOM_FOREST

    FeatureSelectionPipeline.run(sparkSession, trainFile, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByWildcard(featuresPath)
    val selectedTrain: DataFrame = DataService.getDataFromFile(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByWildcard(featuresPath)
    val selectedTest: DataFrame = DataService.getDataFromFile(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, outputTest, metricsPath, modelsPath, classificationMethod)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val metrics: DataFrame = DataService.getDataFromFile(sparkSession, metricsFile)
    assert(metrics.columns.length == 1)
    assert(metrics.count() == 1)

    val modelFile: String = TestUtils.findFileByWildcard(modelsPath)
    assert(modelFile.nonEmpty)
  }
}
