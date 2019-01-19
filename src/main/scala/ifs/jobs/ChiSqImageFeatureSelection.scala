package ifs.jobs

import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

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

  def trainWithCrossValidation(data: DataFrame, featuresColumn: String, labelColumn: String): MLWritable = {
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

  def train(data: DataFrame, featuresColumn: String, labelColumn: String): MLWritable = {
    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
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

  def runPipeline(session: SparkSession, fileRoute: String): MLWritable = {
    val featuresColumn = "features"
    val labelColumn = "output_label"
    val selectedFeaturesColumn = "selected_features"

    var data = getDataFromFile(fileRoute, session, featuresColumn, labelColumn)
    data = selectFeatures(data, session, featuresColumn, labelColumn, selectedFeaturesColumn)

    train(data, selectedFeaturesColumn, labelColumn)
  }

  val Array(featuresFile: String, modelSaveRoute: String) = args
  val sparkSession: SparkSession = SparkSession.builder().appName("ChiSqFeatureSelection").getOrCreate()

  val model: MLWritable = runPipeline(sparkSession, featuresFile)
  model.save(modelSaveRoute)

  sparkSession.stop()
}
