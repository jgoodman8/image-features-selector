package ifs.services

import ifs.utils.ConfigurationService
import org.apache.spark.ml.feature.{MinMaxScaler, QuantileDiscretizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.linalg.Vector

object DataService {
  def getDataFromFile(sparkSession: SparkSession, fileRoute: String): DataFrame = {

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileRoute)
  }

  def saveData(data: DataFrame, file: String, label: String, features: String): Unit = {
    data
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(file)
  }

  def preprocessData(data: DataFrame, label: String, features: String): DataFrame = {

    val labelInput = data.columns.last

    val indexed = this.indexData(data, labelInput, label)
    val scaled = this.scaleFeatures(indexed, label, features)

    scaled
  }

  def preprocessAndDiscretize(data: DataFrame, label: String, features: String): DataFrame = {

    val labelInput = data.columns.last

    val indexed = this.indexData(data, labelInput, label)
    val discretized = this.discretizeFeatures(indexed)
    val assembled = this.assembleFeatures(discretized, label, features)

    assembled
  }

  def splitData(data: DataFrame): Array[DataFrame] = {
    data.randomSplit(Array(ConfigurationService.Data.getTrainSplitRatio, ConfigurationService.Data.getTestSplitRatio))
  }

  private def indexData(data: DataFrame, labelInput: String, labelsOutput: String): DataFrame = {

    val indexedData = new StringIndexer()
      .setInputCol(labelInput)
      .setOutputCol(labelsOutput)
      .fit(data)
      .transform(data)

    indexedData.drop(labelInput)
  }

  private def discretizeFeatures(data: DataFrame): DataFrame = {
    val features = data.columns.dropRight(1)

    var discreteData: DataFrame = data

    features.foreach(feature => {
      discreteData = this.discretizeColumn(discreteData, feature)
    })

    discreteData.drop(features: _*)
  }

  private def discretizeColumn(data: DataFrame, column: String): DataFrame = {
    new QuantileDiscretizer()
      .setInputCol(column)
      .setOutputCol("discrete_" + column)
      .setNumBuckets(10)
      .fit(data)
      .transform(data)
  }

  private def assembleFeatures(data: DataFrame, label: String, assembledFeatures: String): DataFrame = {
    val featuresInput = data.columns.filter(i => !i.equals(label))
    val assembledData = new VectorAssembler()
      .setInputCols(featuresInput)
      .setOutputCol(assembledFeatures)
      .transform(data)

    assembledData.drop(featuresInput: _*)
  }

  private def scaleFeatures(data: DataFrame, label: String, featuresOutput: String,
                            assembledFeatures: String = "assembledFeatures"): DataFrame = {

    val assembledData = this.assembleFeatures(data, label, assembledFeatures)

    val scaledData = new MinMaxScaler()
      .setInputCol(assembledFeatures)
      .setOutputCol(featuresOutput)
      .fit(assembledData)
      .transform(assembledData)

    scaledData.drop(assembledFeatures)
  }

  def extractDenseRows(data: DataFrame, features: String, labels: String): DataFrame = {

    val columnsSize = data.first.getAs[Vector](features).size + 1

    val vecToSeq = udf((v: Vector, label: Double) => v.toArray :+ label)
    val columns = (0 until columnsSize).map(i => col("_tmp").getItem(i).alias(s"f$i"))

    data
      .select(vecToSeq(col(features), col(labels)).alias("_tmp"))
      .select(columns: _*)
  }
}
