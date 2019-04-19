package ifs.services

import ifs.services.ConfigurationService.Preprocess
import ifs.services.ConfigurationService.Preprocess.{Discretize, Scale}
import org.apache.spark.ml.feature.{MinMaxScaler, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

object PreprocessService {

  /**
    * Preprocess the training and testing datasets by performing: label string indexing,
    * features assembling and features standardizing.
    *
    * @param train    Training data set
    * @param test     Testing data set
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def preprocessData(train: DataFrame, test: DataFrame, label: String, features: String): Array[DataFrame] = {

    val labelInput = train.columns.last
    val Array(indexedTrain, indexedTest) = this.indexData(train, test, labelInput, label)
    val Array(scaledTrain, scaledTest) = this.standardizeFeatures(indexedTrain, indexedTest, label, features)

    Array(scaledTrain, scaledTest)
  }

  /**
    * Preprocess the training and testing datasets by performing: label string indexing,
    * features assembling and features scaling.
    *
    * @param train    Training data set
    * @param test     Testing data set
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def preprocessAndScaleData(train: DataFrame, test: DataFrame, label: String, features: String,
                             assembledFeatures: String = "assembledFeatures"): Array[DataFrame] = {

    val labelInput = train.columns.last
    val Array(indexedTrain, indexedTest) = this.indexData(train, test, labelInput, label)

    val assembledTrain = this.assembleFeatures(indexedTrain, label, assembledFeatures)
    val scaledTrain = this.scaleFeatures(assembledTrain, assembledFeatures, features)

    val assembledTest = this.assembleFeatures(indexedTest, label, assembledFeatures)
    val scaledTest = this.scaleFeatures(assembledTest, assembledFeatures, features)

    Array(scaledTrain, scaledTest)
  }

  /**
    * Preprocess the training and testing datasets by performing: label string indexing,
    * features assembling, features min-max scaling and features discretization.
    *
    * @param train    Training data set
    * @param test     Testing data set
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def preprocessAndDiscretize(train: DataFrame, test: DataFrame, label: String, features: String): Array[DataFrame] = {

    val labelInput = train.columns.last
    val Array(indexedTrain, indexedTest) = this.indexData(train, test, labelInput, label)

    val discretizedTrain = this.discretizeFeatures(indexedTrain, label, features)
    val discretizedTest = this.discretizeFeatures(indexedTest, label, features)

    Array(discretizedTrain, discretizedTest)
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

  private def discretizeFeatures(data: DataFrame, label: String, featuresOutput: String,
                                 assembledFeatures: String = "assembledFeatures",
                                 scaledFeatures: String = "scaledFeatures",
                                 numberOfBeans: Int = Discretize.getNumberOfBeans): DataFrame = {

    val assembledData = this.assembleFeatures(data, label, assembledFeatures)
    val scaledData = this.scaleFeatures(assembledData, assembledFeatures, scaledFeatures)

    val discretized = scaledData.rdd.map(this.discretizeRow(_, label, scaledFeatures, numberOfBeans))

    val schema = new StructType()
      .add(StructField(label, DoubleType))
      .add(StructField(featuresOutput, VectorType))

    data.sparkSession.createDataFrame(discretized, schema)
  }

  private def discretizeRow(row: Row, label: String, features: String, numberOfBeans: Int = 3): Row = {
    val rowLabel = row.getAs[Double](label)
    val rowFeatures = row.getAs[DenseVector](features).toArray.map(feature => (feature / numberOfBeans).floor)

    Row(rowLabel, Vectors.dense(rowFeatures))
  }

  private def assembleFeatures(data: DataFrame, label: String, assembledFeatures: String): DataFrame = {
    val featuresInput = data.columns.filter(i => !i.equals(label))
    val assembledData = new VectorAssembler()
      .setInputCols(featuresInput)
      .setOutputCol(assembledFeatures)
      .transform(data)

    assembledData.drop(featuresInput: _*)
  }

  private def scaleFeatures(data: DataFrame, inputFeatures: String, outputFeatures: String): DataFrame = {
    val scaler = new MinMaxScaler()
      .setMin(Scale.getMinScaler)
      .setMax(Scale.getMaxScaler)
      .setInputCol(inputFeatures)
      .setOutputCol(outputFeatures)
      .fit(data)

    scaler.transform(data).drop(inputFeatures)
  }

  private def standardizeFeatures(train: DataFrame, test: DataFrame, label: String, featuresOutput: String,
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
