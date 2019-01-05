package ifs.jobs

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, VectorAssembler}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.{DataFrame, SparkSession}

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
    var dataFrame = new VectorAssembler()
      .setInputCols(data.columns.dropRight(1))
      .setOutputCol(featuresOutput)
      .transform(data)

    dataFrame = dataFrame.withColumn(labelsOutput, data.col(data.columns.last))

    dataFrame
  }

  def train(data: DataFrame): Unit = {
    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val crossValidator = new CrossValidator()
      .setEstimator(logisticRegression)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setNumFolds(5)

    val model = crossValidator.fit(data)

    println(model.avgMetrics)
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

  def runPipeline(session: SparkSession): Unit = {
    val featuresColumn = "features"
    val labelColumn = "label"
    val selectedFeaturesColumn = "selected_features"

    val data = getDataFromFile(featuresFile, sparkSession, featuresColumn, labelColumn)
    val selectedData = selectFeatures(data, sparkSession, featuresColumn, labelColumn, selectedFeaturesColumn)

    train(selectedData)
  }

  val Array(featuresFile) = args
  val sparkSession: SparkSession = SparkSession.builder().appName("ChiSqFeatureSelection").getOrCreate()

  runPipeline(sparkSession)
}
