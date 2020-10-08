package ifs.unit.jobs

import ifs.Constants.Selectors
import ifs.TestUtils
import ifs.jobs.FeatureSelectionPipeline
import ifs.services.DataService
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FeatureSelectionPipelineTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val trainFile: String = TestUtils.getTestDataRoute
  val validationFile: String = TestUtils.getTestDataRoute
  val testFile: String = TestUtils.getTestDataRoute
  val outputPath: String = TestUtils.getFeaturesOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
    TestUtils.clearDirectory(outputPath)
  }

  "runFeatureSelectionPipeline" should "select the best features using the ChiSq method" in {
    val numFeatures = 2
    FeatureSelectionPipeline.run(sparkSession, trainFile, null, testFile, outputPath, Selectors.CHI_SQ, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(outputPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(outputPath, pattern = "test")
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)
  }

  it should "select the best features using the mRMR" in {
    val numFeatures = 2
    FeatureSelectionPipeline.run(sparkSession, trainFile, null, testFile, outputPath, Selectors.MRMR, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(outputPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(outputPath, pattern = "test")
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)
  }

  it should "select the best features using the RELIEF" in {
    val numFeatures = 2
    FeatureSelectionPipeline.run(sparkSession, trainFile, null, testFile, outputPath, Selectors.RELIEF, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(outputPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(outputPath, pattern = "test")
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)
  }

  it should "select the best features using the PCA method" in {
    val numFeatures = 2
    FeatureSelectionPipeline.run(sparkSession, trainFile, null, testFile, outputPath, Selectors.PCA, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(outputPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(outputPath, pattern = "test")
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)
  }

  it should "select the best features from train, validation and test datasets" in {
    val numFeatures = 2
    FeatureSelectionPipeline
      .run(sparkSession, trainFile, validationFile, testFile, outputPath, Selectors.PCA, numFeatures)

    val outputTrain: String = TestUtils.findFileByPattern(outputPath, pattern = "train")
    val selectedTrain: DataFrame = DataService.load(sparkSession, outputTrain)
    assert(selectedTrain.columns.length == numFeatures + 1)

    val outputValidation: String = TestUtils.findFileByPattern(outputPath, pattern = "val")
    val selectedValidation: DataFrame = DataService.load(sparkSession, outputValidation)
    assert(selectedValidation.columns.length == numFeatures + 1)

    val outputTest: String = TestUtils.findFileByPattern(outputPath, pattern = "test")
    val selectedTest: DataFrame = DataService.load(sparkSession, outputTest)
    assert(selectedTest.columns.length == numFeatures + 1)
  }
}
