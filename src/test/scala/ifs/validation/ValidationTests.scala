package ifs.validation

import ifs.Constants.{Classifiers, Selectors}
import ifs.TestUtils
import ifs.jobs.{ClassificationPipeline, FeatureSelectionPipeline}
import ifs.services.DataService
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ValidationTests extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val trainFile: String = TestUtils.getTestDataRoute
  val validationFile: String = TestUtils.getTestDataRoute
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
    val featureSelectionMethod = Selectors.CHI_SQ
    val classificationMethod = Classifiers.LOGISTIC_REGRESSION

    FeatureSelectionPipeline
      .run(sparkSession, trainFile, null, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath, pattern = "test")
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }

  it should "perform the full pipeline (mRMR Selection + Random Forest)" in {
    val numFeatures = 3
    val featureSelectionMethod = Selectors.MRMR
    val classificationMethod = Classifiers.RANDOM_FOREST

    FeatureSelectionPipeline
      .run(sparkSession, trainFile, null, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }

  it should "perform the full pipeline (RELIEF Selection + Random Forest)" in {
    val numFeatures = 3
    val featureSelectionMethod = Selectors.RELIEF
    val classificationMethod = Classifiers.RANDOM_FOREST

    FeatureSelectionPipeline
      .run(sparkSession, trainFile, null, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }

  it should "perform the full pipeline (RELIEF Selection + Naive Bayes)" in {
    val numFeatures = 3
    val featureSelectionMethod = Selectors.RELIEF
    val classificationMethod = Classifiers.NAIVE_BAYES

    FeatureSelectionPipeline
      .run(sparkSession, trainFile, null, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }

  it should "perform the full pipeline (PCA Feature Extraction + Naive Bayes)" in {
    val numFeatures = 3
    val featureSelectionMethod = Selectors.PCA
    val classificationMethod = Classifiers.NAIVE_BAYES

    FeatureSelectionPipeline
      .run(sparkSession, trainFile, null, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }

  it should "perform the full pipeline (RELIEF Selection + Random Forest) using a validation set" in {
    val numFeatures = 3
    val featureSelectionMethod = Selectors.RELIEF
    val classificationMethod = Classifiers.RANDOM_FOREST

    FeatureSelectionPipeline
      .run(sparkSession, trainFile, validationFile, testFile, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputValidation: String = TestUtils.findFileByPattern(featuresPath)
    val selectedValidation: DataFrame = DataService.load(sparkSession, outputValidation)
    assert(selectedValidation.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }

  it should "perform the full pipeline (RELIEF Selection + Random Forest) using a validation set in flowers dataset" in {
    val numFeatures = 3
    val featureSelectionMethod = Selectors.RELIEF
    val classificationMethod = Classifiers.RANDOM_FOREST

    val train = "data/iris/train.csv"
    val validation = "data/iris/val.csv"
    val test = "data/iris/test.csv"

    FeatureSelectionPipeline
      .run(sparkSession, train, validation, test, featuresPath, featureSelectionMethod, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputValidation: String = TestUtils.findFileByPattern(featuresPath)
    val selectedValidation: DataFrame = DataService.load(sparkSession, outputValidation)
    assert(selectedValidation.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(featuresPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)

    ClassificationPipeline.run(sparkSession, outputTrain, null, outputTest, metricsPath, modelsPath, classificationMethod)

    TestUtils.checkMetricsFile(filePattern = "train", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkMetricsFile(filePattern = "test", classificationMethod, metricsPath, sparkSession)
    TestUtils.checkModelPath(modelsPath)
  }
}
