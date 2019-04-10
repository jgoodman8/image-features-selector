package ifs.services

import org.apache.spark.ml.feature.{QuantileDiscretizer, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

object PreprocessService {

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

    val scaler = new StandardScaler()
      .setInputCol(assembledFeatures)
      .setOutputCol(featuresOutput)
      .setWithMean(true)
      .setWithStd(true)
      .fit(assembledTrain)

    val scaledTrain = scaler.transform(assembledTrain).drop(assembledFeatures)
    val scaledTest = scaler.transform(assembledTest).drop(assembledFeatures)

    Array(scaledTrain, scaledTest)
  }
}
