package ifs.jobs

import ifs.services.DataService
import ifs.utils.ConfigurationService
import ifs.Constants
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, SparkSession}


object ClassificationPipeline extends App with Logging {

  def saveMetrics(session: SparkSession, metricNames: String, metrics: Double, outputFolder: String): Unit = {

    import session.implicits._

    val csvRoute = outputFolder + System.currentTimeMillis().toString

    val metricsDF = Seq(metrics).toDF(metricNames)

    metricsDF.write.csv(csvRoute)
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

  def fitWithCrossValidation(data: DataFrame, featuresColumn: String, labelColumn: String): MLWritable = {
    val logisticRegression = new LogisticRegression()
      .setMaxIter(ConfigurationService.Model.getMaxIter)
      .setElasticNetParam(ConfigurationService.Model.getElasticNetParam)
      .setRegParam(ConfigurationService.Model.getRegParam)
      .setFeaturesCol(featuresColumn)
      .setLabelCol(labelColumn)

    val paramGrid = new ParamGridBuilder()
      .addGrid(logisticRegression.regParam, Array(0.2, 0.3))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(logisticRegression)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(labelColumn))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val crossValidatorModel = crossValidator.fit(data)

    crossValidatorModel.avgMetrics.foreach((metric: Double) => {
      logInfo(metric.toString)
    })

    val model: LogisticRegressionModel = crossValidatorModel.bestModel match {
      case m: LogisticRegressionModel => m
      case _ => throw new Exception("Unexpected model type")
    }

    model
  }

  private def fitWithLogisticRegression(data: DataFrame, features: String, label: String): OneVsRestModel = {

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

  private def fitWithRandomForest(data: DataFrame, features: String, label: String): RandomForestClassificationModel = {
    new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
  }

  def fit(data: DataFrame, featuresColumn: String, labelColumn: String, method: String): Model[_] = {
    method match {
      case Constants.LOGISTIC_REGRESSION => this.fitWithLogisticRegression(data, featuresColumn, labelColumn)
      case Constants.RANDOM_FOREST => this.fitWithRandomForest(data, featuresColumn, labelColumn)
    }
  }

  def runTrainPipeline(session: SparkSession, inputFile: String, metricsPath: String, modelsPath: String,
                       method: String, features: String = "features", label: String = "output_label"): Unit = {

    var data = DataService.getDataFromFile(session, inputFile)
    data = DataService.preprocessData(data, features, label)

    val Array(train: DataFrame, test: DataFrame) = DataService.splitData(data)

    val model = this.fit(train, features, label, method)

    this.evaluateAndStoreMetrics(session, model, test, label, metricsPath)
  }


  val Array(appName: String, inputFile: String, method: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  val modelsPath: String = ConfigurationService.Session.getModelPath
  val metricsPath: String = ConfigurationService.Session.getMetricsPath

  this.runTrainPipeline(sparkSession, inputFile, metricsPath, modelsPath, method)


}
