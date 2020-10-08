package ifs.jobs

import ifs.Constants.Classifiers._
import ifs.services._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.sql.{DataFrame, SparkSession}


object ClassificationPipeline extends App with Logging {

  def fit(trainData: DataFrame,
          validationData: DataFrame,
          label: String,
          features: String,
          method: String,
          modelPath: String): Model[_] = {
    val model = method match {
      case LOGISTIC_REGRESSION => ModelService.fitWithLogisticRegression(trainData, validationData, label, features)
      case RANDOM_FOREST => ModelService.fitWithRandomForest(trainData, validationData, label, features)
      case DECISION_TREE => ModelService.fitWithDecisionTree(trainData, validationData, label, features)
      case MLP => ModelService.fitWithMLP(trainData, validationData, label, features)
      case NAIVE_BAYES => ModelService.fitWithNaiveBayes(trainData, label, features)
      case SVM => ModelService.fitWithSVM(trainData, validationData, label, features)
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

    ClassificationService.saveMetrics(session, metricNames, trainMetricValues, metricsPath + "/train")
    ClassificationService.saveMetrics(session, metricNames, testMetricValues, metricsPath + "/test")
  }

  def preprocess(datasets: Array[DataFrame],
                 label: String,
                 features: String,
                 method: String): Array[DataFrame] = {
    method match {
      case LOGISTIC_REGRESSION | RANDOM_FOREST | DECISION_TREE | MLP => PreprocessService
        .preprocessData(datasets, label, features)
      case NAIVE_BAYES | SVM => PreprocessService.preprocessAndScaleData(datasets, label, features)
      case _ => throw new NoSuchMethodException("The classifier method is not implemented")
    }
  }

  def run(session: SparkSession,
          trainFile: String,
          validationFile: String,
          testFile: String,
          metricsPath: String,
          modelPath: String,
          method: String,
          features: String = "features",
          label: String = "output_label"): Unit = {

    val hasValidationData = validationFile != null
    val datasets = PipelineUtils.load(session, trainFile, validationFile, testFile, hasValidationData)
    val preprocessedDatasets = this.preprocess(datasets, label, features, method)

    if (hasValidationData) {
      val model = this.fit(preprocessedDatasets(0), preprocessedDatasets(1), label, features, method, modelPath)
      this.evaluate(session, model, preprocessedDatasets(0), preprocessedDatasets(2), label, metricsPath, method)
    } else {
      val model = this.fit(preprocessedDatasets(0), null, label, features, method, modelPath)
      this.evaluate(session, model, preprocessedDatasets(0), preprocessedDatasets(1), label, metricsPath, method)
    }
  }

  val appName = args(0)
  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  val metricsPath: String = ConfigurationService.Session.getMetricsPath
  val modelsPath: String = ConfigurationService.Session.getModelPath

  val sizeWithValidationFile = 5
  val sizeWithNoValidationFile = 4
  if (args.length == sizeWithValidationFile) {
    val Array(_, trainFile: String, validationFile: String, testFile: String, method: String) = args
    this.run(sparkSession, trainFile, validationFile, testFile, metricsPath, modelsPath, method)
  } else if (args.length == sizeWithNoValidationFile) {
    val Array(_, trainFile: String, testFile: String, method: String) = args
    this.run(sparkSession, trainFile, null, testFile, metricsPath, modelsPath, method)
  }

  sparkSession.stop()
}
