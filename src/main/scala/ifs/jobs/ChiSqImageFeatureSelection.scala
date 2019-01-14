package ifs.jobs

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

object ChiSqImageFeatureSelection extends App {

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

  def train(data: DataFrame, featuresColumn: String, labelColumn: String): Unit = {
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

    val model = crossValidator.fit(data)

    model.avgMetrics.foreach(metric => println(metric)) // TODO: remove and return the best model
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

  def runPipeline(session: SparkSession, fileRoute: String): Unit = {
    val featuresColumn = "features"
    val labelColumn = "output_label"
    val selectedFeaturesColumn = "selected_features"

    var data = getDataFromFile(fileRoute, session, featuresColumn, labelColumn)
    data = selectFeatures(data, session, featuresColumn, labelColumn, selectedFeaturesColumn)

    train(data, selectedFeaturesColumn, labelColumn)
  }

  val Array(featuresFile) = args
  val sparkSession: SparkSession = SparkSession.builder().appName("ChiSqFeatureSelection").getOrCreate()

  runPipeline(sparkSession, featuresFile)
}
