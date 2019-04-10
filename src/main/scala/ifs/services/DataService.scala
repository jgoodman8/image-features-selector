package ifs.services

import ifs.utils.ConfigurationService
import org.apache.spark.ml.feature.{MinMaxScaler, QuantileDiscretizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object DataService {
  def getDataFromFile(sparkSession: SparkSession, fileRoute: String): DataFrame = {

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileRoute)
  }

  def saveData(data: DataFrame, fileDir: String, label: String, features: String): Unit = {
    data
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(fileDir + System.currentTimeMillis.toString)
  }

  def preprocessData(train: DataFrame, test: DataFrame, label: String, features: String): Array[DataFrame] = {
    val labelInput = train.columns.last

    val Array(indexedTrain, indexedTest) = this.indexData(train, test, labelInput, label)
    val Array(scaledTrain, scaledTest) = this.scaleFeatures(indexedTrain, indexedTest, label, features)

    Array(scaledTrain, scaledTest)
  }

  def preprocessAndDiscretize(train: DataFrame, test: DataFrame, label: String, features: String): Array[DataFrame] = {

    val labelInput = train.columns.last

    val Array(indexedTrain, indexedTest) = this.indexData(train, test, labelInput, label)
    val discretizedTrain = this.discretizeFeatures(indexedTrain)
    val discretizedTest = this.discretizeFeatures(indexedTest)

    val assembledTrain = this.assembleFeatures(discretizedTrain, label, features)
    val assembledTest = this.assembleFeatures(discretizedTest, label, features)

    Array(assembledTrain, assembledTest)
  }

  def splitData(data: DataFrame): Array[DataFrame] = {
    data.randomSplit(Array(ConfigurationService.Data.getTrainSplitRatio, ConfigurationService.Data.getTestSplitRatio))
  }

  private def indexData(train: DataFrame, test: DataFrame, inputCol: String, outputCol: String): Array[DataFrame] = {

    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .fit(train)

    val indexedTrain = indexer.transform(train).drop(inputCol)
    val indexedTest = indexer.transform(test).drop(inputCol)

    Array(indexedTrain, indexedTest)
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

  private def scaleFeatures(train: DataFrame, test: DataFrame, label: String, featuresOutput: String,
                            assembledFeatures: String = "assembledFeatures"): Array[DataFrame] = {

    val assembledTrain = this.assembleFeatures(train, label, assembledFeatures)
    val assembledTest = this.assembleFeatures(test, label, assembledFeatures)

    val scaler = new MinMaxScaler()
      .setInputCol(assembledFeatures)
      .setOutputCol(featuresOutput)
      .fit(assembledTrain)

    val scaledTrain = scaler.transform(assembledTrain).drop(assembledFeatures)
    val scaledTest = scaler.transform(assembledTest).drop(assembledFeatures)

    Array(scaledTrain, scaledTest)
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
