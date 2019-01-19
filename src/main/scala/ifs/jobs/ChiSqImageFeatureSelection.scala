package ifs.jobs

import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer

object ChiSqImageFeatureSelection extends App with Logging {

  def getDataFromFile(fileRoute: String,
                      sparkSession: SparkSession,
                      featuresOutput: String,
                      labelsOutput: String): DataFrame = {

    val data = sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileRoute)

    toDenseDF(data, featuresOutput, labelsOutput)
  }

  def toDenseDF(data: DataFrame, featuresOutput: String, labelsOutput: String): DataFrame = {
    var dataFrame = new StringIndexer()
      .setInputCol(data.columns.last)
      .setOutputCol(labelsOutput)
      .fit(data)
      .transform(data)

    dataFrame = dataFrame.withColumn(labelsOutput, col(labelsOutput).cast(DoubleType))

    dataFrame = new VectorAssembler()
      .setInputCols(data.columns.dropRight(1))
      .setOutputCol(featuresOutput)
      .transform(dataFrame)

    dataFrame
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
      .setMaxIter(100)
      .setElasticNetParam(0.8)
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

    val metricNames = Array("f1", "weightedPrecision", "weightedRecall", "accuracy")
    val metrics: ArrayBuffer[Double] = new ArrayBuffer[Double]()


    metricNames.foreach(metricName => {
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(labelColumn)
        .setPredictionCol("prediction")
        .setMetricName(metricName)


      val predictions = model.transform(test)
      val metric = evaluator.evaluate(predictions)

      metrics.append(metric)
    })

    saveMetrics(session, metricNames, metrics.toArray, outputFolder)
  }

  def saveMetrics(session: SparkSession,
                  metricNames: Array[String],
                  metrics: Array[Double],
                  outputFolder: String): Unit = {

    import session.implicits._

    val csvRoute = outputFolder + System.currentTimeMillis().toString + ".csv"

    val metricsDF = Seq(
      (metrics(0), metrics(1), metrics(2), metrics(3))
    ).toDF(metricNames: _*)

    metricsDF.write.csv(csvRoute)
  }

  def runPipeline(session: SparkSession, fileRoute: String, outputFolder: String): MLWritable = {
    val featuresColumn = "features"
    val labelColumn = "output_label"
    val selectedFeaturesColumn = "selected_features"

    var data = getDataFromFile(fileRoute, session, featuresColumn, labelColumn)
    data = selectFeatures(data, session, featuresColumn, labelColumn, selectedFeaturesColumn)

    val Array(train: DataFrame, test: DataFrame) = data.randomSplit(Array(0.7, 0.3))

    val model = fit(train, selectedFeaturesColumn, labelColumn)

    evaluateAndStoreMetrics(session, model, test, labelColumn, outputFolder)

    model
  }

  val appName = "ChiSqFeatureSelection"

  val Array(featuresFile: String, modelSaveRoute: String, outputFolder: String) = args
  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  val model: MLWritable = runPipeline(sparkSession, featuresFile, outputFolder)
  model.write.overwrite().save(modelSaveRoute)

  sparkSession.stop()
}
