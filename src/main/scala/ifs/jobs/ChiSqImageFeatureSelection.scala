package ifs.jobs

import ifs.utils.ConfigurationService
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel

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

  def fit(data: DataFrame, featuresColumn: String, labelColumn: String): RandomForestClassificationModel = {
    val logisticRegression = new RandomForestClassifier()
      .setFeaturesCol(featuresColumn)
      .setLabelCol(labelColumn)

    logisticRegression.fit(data)
  }

  def selectFeatures(data: DataFrame,
                     sparkSession: SparkSession,
                     features: String,
                     labels: String,
                     selectedFeatures: String,
                     numTopFeatures: Int = 10): DataFrame = {

    val selector = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(labels)
      .setOutputCol(selectedFeatures)

    selector
      .fit(data)
      .transform(data)
      .select(selectedFeatures, labels)
  }

  def evaluateAndStoreMetrics(session: SparkSession,
                              model: RandomForestClassificationModel,
                              test: DataFrame,
                              labelColumn: String,
                              outputFolder: String,
                              metricName: String = "accuracy"): Unit = {
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

  private def extractDenseRows(data: DataFrame, featuresColumn: String): DataFrame = {
    val featuresSize = data.first.getAs[org.apache.spark.mllib.linalg.Vector](featuresColumn).size

    // Simple helper to convert vector to array<double>
    val vecToSeq = udf((v: Vector) => v.toArray)

    // Prepare a list of columns to create
    val exprs = (0 until featuresSize).map(i => col("_tmp").getItem(i).alias(s"f$i"))

    data
      .select(vecToSeq(col(featuresColumn)).alias("_tmp"))
      .select(exprs: _*)
  }

  def runFeatureSelectionPipeline(session: SparkSession, inputFile: String, outputFile: String,
                                  featuresColumn: String = "features", labelColumn: String = "output_label",
                                  selectedFeaturesColumn: String = "selected_features"): Unit = {

    val data = preprocessData(getDataFromFile(inputFile, session), featuresColumn, labelColumn)
    data.persist(StorageLevel.MEMORY_AND_DISK_SER)

    val selectedData = selectFeatures(data, session, featuresColumn, labelColumn, selectedFeaturesColumn)
    data.unpersist()
    selectedData.persist(StorageLevel.MEMORY_AND_DISK_SER)

    extractDenseRows(selectedData, selectedFeaturesColumn)
      .withColumn(labelColumn, selectedData(labelColumn))
      .write.csv(outputFile)

    selectedData.unpersist()
  }

  def runTrainPipeline(session: SparkSession, inputFile: String, metricsPath: String, modelsPath: String,
                       featuresColumn: String = "features", labelColumn: String = "output_label"): Unit = {

    val data = preprocessData(getDataFromFile(inputFile, session), featuresColumn, labelColumn)

    val Array(train: DataFrame, test: DataFrame) = data.randomSplit(getSplitData)

    val model = fit(train, featuresColumn, labelColumn)

    evaluateAndStoreMetrics(session, model, test, labelColumn, metricsPath)

    model.write.overwrite().save(modelsPath)
  }

  val Array(appName: String, inputFile: String, outputFile: String, method: String) = args

  val sparkSession: SparkSession = SparkSession.builder()
    .appName(appName)
    .config("spark.driver.maxResultSize", ConfigurationService.Session.getDriverMaxResultSize)
    .getOrCreate()

  sparkSession.sparkContext.setCheckpointDir(ConfigurationService.Session.getCheckpointDir)

  val modelsPath: String = ConfigurationService.Session.getModelPath
  val metricsPath: String = ConfigurationService.Session.getMetricsPath

  if (method == "chisq") {
    runFeatureSelectionPipeline(sparkSession, inputFile, outputFile)
  } else if (method == "train") {
    runTrainPipeline(sparkSession, inputFile, metricsPath, modelsPath)
  }

  sparkSession.stop()
}
