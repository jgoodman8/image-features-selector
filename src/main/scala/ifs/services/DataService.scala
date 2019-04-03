package ifs.services

import ifs.utils.ConfigurationService
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataService {
  def getDataFromFile(sparkSession: SparkSession, fileRoute: String): DataFrame = {

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileRoute)
  }

  def preprocessData(data: DataFrame, featuresOutput: String, labelsOutput: String): DataFrame = {

    var preprocessedData = this.preprocessLabels(data, labelsOutput)
    preprocessedData = this.preprocessFeatures(preprocessedData, featuresOutput)

    preprocessedData
  }

  def preprocessLabels(data: DataFrame, labelsOutput: String): DataFrame = {

    val labelsInput = data.columns.last
    val indexedData = new StringIndexer()
      .setInputCol(labelsInput)
      .setOutputCol(labelsOutput)
      .fit(data)
      .transform(data)

    indexedData.drop(labelsInput)
  }

  def preprocessFeatures(data: DataFrame, featuresOutput: String,
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
      .drop(featuresInput: _*)
      .drop(assembledFeatures)
  }

  def splitData(data: DataFrame): Array[DataFrame] = {
    data.randomSplit(Array(ConfigurationService.Data.getTrainSplitRatio, ConfigurationService.Data.getTestSplitRatio))
  }
}
