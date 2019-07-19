package ifs.jobs


import ifs.services.{ClassificationService, ConfigurationService, DataService}
import ifs.strategies.{EvaluationStrategy, FitStrategy, PreprocessStrategy}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.sql.{DataFrame, SparkSession}


object ClassificationPipeline extends App with Logging {

  def fit(data: DataFrame, label: String, features: String, method: String, modelPath: String): Model[_] = {
    val model = FitStrategy.fit_with_strategy(data, label, features, method)
    model.write.save(modelPath + "/" + System.currentTimeMillis.toString)
    model
  }

  def evaluate(session: SparkSession, model: Model[_], train: DataFrame, test: DataFrame, label: String,
               metricsPath: String, method: String): Unit = {

    val (metricNames, trainMetricValues, testMetricValues) = EvaluationStrategy
      .evaluate_with_strategy(method, model, train, test, label)

    ClassificationService.saveMetrics(session, metricNames, trainMetricValues, metricsPath + "/train_eval_")
    ClassificationService.saveMetrics(session, metricNames, testMetricValues, metricsPath + "/test_eval_")
  }

  def run(session: SparkSession, trainFile: String, testFile: String, metricsPath: String, modelPath: String,
          method: String, features: String = "features", label: String = "output_label"): Unit = {

    val train: DataFrame = DataService.load(session, trainFile)
    val test: DataFrame = DataService.load(session, testFile)

    val Array(preprocessedTrain, preprocessedTest) = PreprocessStrategy
      .preprocess_for_training(train, test, label, features, method)

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
