package ifs.jobs

import ifs.Constants
import ifs.services.DataService
import ifs.utils.ConfigurationService
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object ClassificationPipeline extends App with Logging {

  def saveMetrics(session: SparkSession, metricNames: String, metrics: Double, outputFolder: String): Unit = {
    import session.implicits._

    Seq(metrics).toDF(metricNames)
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(outputFolder + "/" + System.currentTimeMillis.toString)
  }

  def evaluateAndStoreMetrics(session: SparkSession, model: Model[_], test: DataFrame, labelCol: String,
                              outputFolder: String, metricName: String = "accuracy",
                              predictionCol: String = "prediction"): Unit = {

    val predictions = model.transform(test).select(labelCol, predictionCol)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predictionCol)
      .setMetricName(metricName)

    val metricValue = evaluator.evaluate(predictions)

    this.saveMetrics(session, metricName, metricValue, outputFolder)
  }

  private def fitWithLogisticRegression(data: DataFrame, label: String, features: String): OneVsRestModel = {

    val logisticRegression = new LogisticRegression()
      .setMaxIter(ConfigurationService.Model.getMaxIter)
      .setElasticNetParam(ConfigurationService.Model.getElasticNetParam)
      .setRegParam(ConfigurationService.Model.getRegParam)
      .setFeaturesCol(features)
      .setLabelCol(label)

    new OneVsRest()
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setClassifier(logisticRegression)
      .fit(data)
  }

  private def fitWithRandomForest(data: DataFrame, label: String, features: String): RandomForestClassificationModel = {
    new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
  }

  def fit(data: DataFrame, label: String, features: String, method: String, modelPath: String): Model[_] = {
    val model = method match {
      case Constants.LOGISTIC_REGRESSION => this.fitWithLogisticRegression(data, label, features)
      case Constants.RANDOM_FOREST => this.fitWithRandomForest(data, label, features)
    }

    model.write.save(modelPath + "/" + System.currentTimeMillis.toString)

    model
  }

  def run(session: SparkSession, trainFile: String, testFile: String, metricsPath: String, modelPath: String,
          method: String, features: String = "features", label: String = "output_label"): Unit = {

    val train: DataFrame = DataService.getDataFromFile(session, trainFile)
    val test: DataFrame = DataService.getDataFromFile(session, testFile)

    val Array(preprocessedTrain, preprocessedTest) = DataService.preprocessData(train, test, label, features)

    val model: Model[_] = this.fit(preprocessedTrain, label, features, method, modelPath)

    this.evaluateAndStoreMetrics(session, model, preprocessedTest, label, metricsPath)
  }

  val Array(appName: String, trainFile: String, testFile: String, method: String) = args

  val sparkSession: SparkSession = SparkSession.builder()
    .appName(name = f"$appName%s_$method%s")
    .getOrCreate()

  val metricsPath: String = ConfigurationService.Session.getMetricsPath
  val modelsPath: String = ConfigurationService.Session.getModelPath

  this.run(sparkSession, trainFile, testFile, metricsPath, modelsPath, method)

  sparkSession.stop()
}
