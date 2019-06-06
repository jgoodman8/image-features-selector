package ifs.jobs

import ifs.Constants.Classifiers._
import ifs.services.{ClassificationService, ConfigurationService, DataService, ModelService, PreprocessService}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel


object ClassificationPipeline extends App with Logging {

  def fit(data: DataFrame, label: String, features: String, method: String, modelPath: String): Model[_] = {
    val model = method match {
      case LOGISTIC_REGRESSION => ModelService.fitWithLogisticRegression(data, label, features)
      case RANDOM_FOREST => ModelService.fitWithRandomForest(data, label, features)
      case DECISION_TREE => ModelService.fitWithDecisionTree(data, label, features)
      case MLP => ModelService.fitWithMLP(data, label, features)
      case NAIVE_BAYES => ModelService.fitWithNaiveBayes(data, label, features)
      case SVM => ModelService.fitWithSVM(data, label, features)
      case _ => throw new NoSuchMethodException("The classifier method is not implemented")
    }

    model.write.save(modelPath + "/" + System.currentTimeMillis.toString)

    model
  }

  def evaluate(session: SparkSession, model: Model[_], train: DataFrame, test: DataFrame, label: String,
               metricsPath: String, method: String): Unit = {

    var metricNames: Array[String] = ConfigurationService.Model.getMetrics
    var trainMetricValues: Array[Double] = ClassificationService.evaluate(model, train, label, metricNames)
    var testMetricValues: Array[Double] = ClassificationService.evaluate(model, test, label, metricNames)

    if (method == NAIVE_BAYES || method == DECISION_TREE || method == RANDOM_FOREST || method == MLP) {
      val trainTopAccuracy = ClassificationService.getTopAccuracyN(model, train, label)
      val testTopAccuracy = ClassificationService.getTopAccuracyN(model, test, label)

      metricNames = metricNames :+ "topNAccuracy"
      trainMetricValues = trainMetricValues :+ trainTopAccuracy
      testMetricValues = testMetricValues :+ testTopAccuracy
    }

    ClassificationService.saveMetrics(session, metricNames, trainMetricValues, metricsPath + "/train_eval_")
    ClassificationService.saveMetrics(session, metricNames, testMetricValues, metricsPath + "/test_eval_")
  }

  def preprocess(train: DataFrame, test: DataFrame, label: String, features: String,
                 method: String): Array[DataFrame] = {
    method match {
      case LOGISTIC_REGRESSION | RANDOM_FOREST | DECISION_TREE | MLP => PreprocessService
        .preprocessData(train, test, label, features)
      case NAIVE_BAYES | SVM => PreprocessService.preprocessAndScaleData(train, test, label, features)
      case _ => throw new NoSuchMethodException("The classifier method is not implemented")
    }
  }

  def run(session: SparkSession, trainFile: String, testFile: String, metricsPath: String, modelPath: String,
          method: String, features: String = "features", label: String = "output_label"): Unit = {

    val train: DataFrame = DataService.load(session, trainFile)
    val test: DataFrame = DataService.load(session, testFile)

    train.persist(StorageLevel.MEMORY_AND_DISK)
    test.persist(StorageLevel.MEMORY_AND_DISK)

    val Array(preprocessedTrain, preprocessedTest) = this.preprocess(train, test, label, features, method)

    train.unpersist(true)
    test.unpersist(true)

    preprocessedTrain.persist(StorageLevel.MEMORY_AND_DISK)
    preprocessedTest.persist(StorageLevel.MEMORY_AND_DISK)

    val model = this.fit(preprocessedTrain, label, features, method, modelPath)

    this.evaluate(session, model, preprocessedTrain, preprocessedTest, label, metricsPath, method)
  }

  val Array(appName: String, trainFile: String, testFile: String, method: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  val metricsPath: String = ConfigurationService.Session.getMetricsPath
  val modelsPath: String = ConfigurationService.Session.getModelPath

  this.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

  sparkSession.stop()
}
