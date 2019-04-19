package ifs.jobs

import ifs.Constants
import ifs.services.{ClassificationService, ConfigurationService, DataService, PreprocessService}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.sql.{DataFrame, SparkSession}


object ClassificationPipeline extends App with Logging {

  def fit(data: DataFrame, label: String, features: String, method: String, modelPath: String): Model[_] = {
    val model = method match {
      case Constants.LOGISTIC_REGRESSION => ClassificationService.fitWithLogisticRegression(data, label, features)
      case Constants.RANDOM_FOREST => ClassificationService.fitWithRandomForest(data, label, features)
      case Constants.DECISION_TREE => ClassificationService.fitWithDecisionTree(data, label, features)
      case Constants.MLP => ClassificationService.fitWithMLP(data, label, features)
    }

    model.write.save(modelPath + "/" + System.currentTimeMillis.toString)

    model
  }

  def evaluate(session: SparkSession, model: Model[_], test: DataFrame, label: String, metricsPath: String): Unit = {
    val metricNames: Array[String] = ConfigurationService.Model.getMetrics
    val metricValues: Array[Double] = ClassificationService.evaluate(model, test, label, metricNames)

    ClassificationService.saveMetrics(session, metricNames, metricValues, metricsPath)
  }

  def run(session: SparkSession, trainFile: String, testFile: String, metricsPath: String, modelPath: String,
          method: String, features: String = "features", label: String = "output_label"): Unit = {

    val train: DataFrame = DataService.load(session, trainFile)
    val test: DataFrame = DataService.load(session, testFile)

    val Array(preprocessedTrain, preprocessedTest) = PreprocessService.preprocessData(train, test, label, features)

    val model = this.fit(preprocessedTrain, label, features, method, modelPath)

    this.evaluate(session, model, preprocessedTest, label, metricsPath)
  }

  val Array(appName: String, trainFile: String, testFile: String, method: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName(name = f"$appName%s_$method%s").getOrCreate()

  val metricsPath: String = ConfigurationService.Session.getMetricsPath
  val modelsPath: String = ConfigurationService.Session.getModelPath

  this.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

  sparkSession.stop()
}
