package ifs.unit.jobs

import ifs.jobs.ClassificationPipeline
import ifs.services.{ConfigurationService, DataService}
import ifs.TestUtils
import ifs.Constants.Classifiers
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
    //    TestUtils.clearDirectory(metricsPath)
  }

  "trainPipeline" should "classify the dataset using a Logistic Regression model" in {
    val method = Classifiers.LOGISTIC_REGRESSION
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    checkMetricsFile(filePattern = "train")
    checkMetricsFile(filePattern = "test")
    checkModelPath()
  }

  it should "classify the dataset using a Random Forest model" in {
    val method = Classifiers.RANDOM_FOREST
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    checkMetricsFile(filePattern = "train")
    checkMetricsFile(filePattern = "test")
    checkModelPath()
  }

  it should "classify the dataset using a Decision Tree Classifier" in {
    val method = Classifiers.DECISION_TREE
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    checkMetricsFile(filePattern = "train")
    checkMetricsFile(filePattern = "test")
    checkModelPath()
  }

  it should "classify the dataset using a MultiLayer Perceptron" in {
    val method = Classifiers.MLP
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    checkMetricsFile(filePattern = "train")
    checkMetricsFile(filePattern = "test")
    checkModelPath()
  }

  it should "classify the dataset using a Naive Bayes Classifier" in {
    val method = Classifiers.NAIVE_BAYES
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    checkMetricsFile(filePattern = "train")
    checkMetricsFile(filePattern = "test")
    checkModelPath()
  }

  it should "classify the dataset using a Support Vector Machine with Linear Kernel" in {
    val method = Classifiers.SVM
    ClassificationPipeline.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

    checkMetricsFile(filePattern = "train")
    checkMetricsFile(filePattern = "test")
    checkModelPath()
  }

  ignore should "classify the BoW features without selecting any feature" in {
    val method = Classifiers.RANDOM_FOREST
    val train = "data/tiny-imagenet-features/tiny_imagenet_bow_k100_surf_train_v2.csv"
    val test = "data/tiny-imagenet-features/tiny_imagenet_bow_k100_surf_val_v2.csv"

    ClassificationPipeline.run(sparkSession, train, test, metricsPath, modelsPath, method)
  }

  def checkMetricsFile(filePattern: String): Unit = {
    val metricsFile: String = TestUtils.findFileByPattern(metricsPath, filePattern)
    val metrics: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(metrics.columns.length == 2)
    assert(metrics.count() == ConfigurationService.Model.getMetrics.length)
  }

  def checkModelPath(): Unit = {
    val modelFile: String = TestUtils.findFileByPattern(modelsPath)
    assert(modelFile.nonEmpty)
  }
}
