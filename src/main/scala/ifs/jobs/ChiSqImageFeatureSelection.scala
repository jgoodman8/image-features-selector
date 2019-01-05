package ifs.jobs

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, VectorAssembler}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

object ChiSqImageFeatureSelection extends App {

  private val featuresColumnName = "features"
  private val labelColumnName = "label"
  private val selectedFeaturesColumnName = "selected_features"

  def getDataFromFile(fileRoute: String, sparkSession: SparkSession): DataFrame = {
    var data = sparkSession.read.format("csv")
      .option("header", "true")
      .load(fileRoute)

    data.columns.foreach((columnName: String) => {
      data = data.withColumn(columnName, col(columnName).cast(DoubleType))
    })

    val assembler = new VectorAssembler()
      .setInputCols(data.columns.dropRight(1))
      .setOutputCol(featuresColumnName)

    var assembledData = assembler.transform(data)
    assembledData = assembledData.withColumn(labelColumnName, data.col(data.columns.last))

    assembledData
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

  def selectFeatures(data: DataFrame, sparkSession: SparkSession): DataFrame = {
    val selector = new ChiSqSelector()
      .setNumTopFeatures(10)
      .setFeaturesCol(featuresColumnName)
      .setLabelCol(labelColumnName)
      .setOutputCol(selectedFeaturesColumnName)

    selector.fit(data).transform(data)
  }

  def runPipeline(session: SparkSession): Unit = {
    val data = getDataFromFile(featuresFile, sparkSession)
    val dataFrameWithSelectedFeatures = selectFeatures(data, sparkSession)
    train(dataFrameWithSelectedFeatures)
  }

  val Array(featuresFile) = args
  val sparkSession: SparkSession = SparkSession.builder().appName("ChiSqFeatureSelection").getOrCreate()

  runPipeline(sparkSession)
}
