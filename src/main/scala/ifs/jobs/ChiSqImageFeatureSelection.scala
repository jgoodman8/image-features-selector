package ifs.jobs

import ifs.utils.ConfigurationService
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

object ChiSqImageFeatureSelection extends App with Logging {

  def getDataFromFile(fileRoute: String,
                      sparkSession: SparkSession): DataFrame = {

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileRoute)
  }

  def preprocessData(data: DataFrame, featuresOutput: String, labelsOutput: String): DataFrame = {
    var preprocessedData = preprocessLabels(data, labelsOutput)
    preprocessedData = preprocessFeatures(preprocessedData, featuresOutput)

    preprocessedData
  }

  private def preprocessLabels(data: DataFrame, labelsOutput: String): DataFrame = {
    val labelsInput = data.columns.last
    val indexedData = new StringIndexer()
      .setInputCol(labelsInput)
      .setOutputCol(labelsOutput)
      .fit(data)
      .transform(data)

    indexedData.withColumn(labelsOutput, col(labelsOutput).cast(DoubleType))
  }

  private def preprocessFeatures(data: DataFrame,
                                 featuresOutput: String,
                                 assembledFeatures: String = "assembledFeatures"): DataFrame = {
    val featuresInput = data.columns.dropRight(1)
    val assembledData = new VectorAssembler()
      .setInputCols(featuresInput)
      .setOutputCol(assembledFeatures)
      .transform(data)

    val scaledData = new MinMaxScaler()
      .setInputCol(assembledFeatures)
      .setOutputCol(featuresOutput)
      .fit(assembledData)
      .transform(assembledData)

    scaledData
  }

  def fitWithCrossValidation(data: DataFrame, featuresColumn: String, labelColumn: String): MLWritable = {
    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)
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

  def fit(data: DataFrame, featuresColumn: String, labelColumn: String): LogisticRegressionModel = {
    val logisticRegression = new LogisticRegression()
      .setMaxIter(ConfigurationService.Model.getMaxIter)
      .setElasticNetParam(ConfigurationService.Model.getElasticNetParam)
      .setRegParam(ConfigurationService.Model.getRegParam)
      .setFeaturesCol(featuresColumn)
      .setLabelCol(labelColumn)

    logisticRegression.fit(data)
  }

  def selectFeatures(data: DataFrame,
                     sparkSession: SparkSession,
                     features: String,
                     labels: String,
                     output: String,
                     featuresSelected: Int = 10): DataFrame = {

    val selector = new ChiSqSelector()
      .setNumTopFeatures(featuresSelected)
      .setFeaturesCol(features)
      .setLabelCol(labels)
      .setOutputCol(output)

    selector.fit(data).transform(data)
  }

  def evaluateAndStoreMetrics(session: SparkSession,
                              model: LogisticRegressionModel,
                              test: DataFrame,
                              labelColumn: String,
                              outputFolder: String): Unit = {

    val metricName = "accuracy"

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setPredictionCol("prediction")
      .setMetricName(metricName)


    val predictions = model.transform(test)
    val metricValue = evaluator.evaluate(predictions)

    saveMetrics(session, metricName, metricValue, outputFolder)
  }

  def getSplitData: Array[Double] = {
    Array(ConfigurationService.Data.getTrainSplitRatio, ConfigurationService.Data.getTestSplitRatio)
  }

  def saveMetrics(session: SparkSession,
                  metricNames: String,
                  metrics: Double,
                  outputFolder: String): Unit = {

    import session.implicits._

    val csvRoute = outputFolder + System.currentTimeMillis().toString

    val metricsDF = Seq(metrics).toDF(metricNames)

    metricsDF.write.csv(csvRoute)
  }

  def runFullPipeline(session: SparkSession, fileRoute: String, outputFolder: String,
                      featuresColumn: String = "features", labelColumn: String = "output_label",
                      selectedFeaturesColumn: String = "selected_features"): MLWritable = {

    var data = getDataFromFile(fileRoute, session)
    data = preprocessData(data, featuresColumn, labelColumn)

    data.persist()

    data = selectFeatures(data, session, featuresColumn, labelColumn, selectedFeaturesColumn)

    val Array(train: DataFrame, test: DataFrame) = data.randomSplit(getSplitData)

    train.persist()
    test.persist()

    val model = fit(train, selectedFeaturesColumn, labelColumn)
    evaluateAndStoreMetrics(session, model, test, labelColumn, outputFolder)

    model
  }

  def runTrainPipeline(session: SparkSession, fileRoute: String, outputFolder: String,
                       featuresColumn: String = "features", labelColumn: String = "output_label"): MLWritable = {

    var data = getDataFromFile(fileRoute, session)
    data = preprocessData(data, featuresColumn, labelColumn)

    val Array(train: DataFrame, test: DataFrame) = data.randomSplit(getSplitData)

    train.persist()
    test.persist()

    val model = fit(train, featuresColumn, labelColumn)
    evaluateAndStoreMetrics(session, model, test, labelColumn, outputFolder)

    model
  }

  val appName = "ChiSqFeatureSelection"

  val Array(featuresFile: String, method: String) = args

  val sparkSession: SparkSession = SparkSession.builder()
    .appName(appName)
    .config("spark.driver.maxResultSize", ConfigurationService.Session.getDriverMaxResultSize)
    .getOrCreate()

  sparkSession.sparkContext.setCheckpointDir(ConfigurationService.Session.getCheckpointDir)

  val modelsPath: String = ConfigurationService.Session.getModelDir
  val outputFolder: String = ConfigurationService.Session.getOutputDir

  val model: MLWritable = if (method == "chisq") {
    runFullPipeline(sparkSession, featuresFile, outputFolder)
  } else if (method == "train") {
    runTrainPipeline(sparkSession, featuresFile, outputFolder)
  } else {
    runTrainPipeline(sparkSession, featuresFile, outputFolder)
  }

  model.write.overwrite().save(modelsPath)

  sparkSession.stop()
}
