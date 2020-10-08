package ifs.services

import ifs.services.ConfigurationService.Preprocess.{Discretize, Scale}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame

object PreprocessService {

  /**
    * Preprocess the training and testing datasets by performing: label string indexing,
    * features assembling and features standardizing.
    *
    * @param datasets List of datasets
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def standardizeData(datasets: Array[DataFrame], label: String, features: String): Array[DataFrame] = {

    val train = datasets(0)
    val labelInput = train.columns.last
    val indexedDatasets = this.indexData(datasets, labelInput, label)
    val standardizedDatasets = this.standardizeFeatures(indexedDatasets, label, features)

    standardizedDatasets
  }

  /**
    * Preprocess the given datasets by performing: label string indexing and features assembling.
    *
    * @param datasets Array of datasets
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def preprocessData(datasets: Array[DataFrame],
                     label: String,
                     features: String): Array[DataFrame] = {

    val labelInput = datasets(0).columns.last
    val indexedSets = this.indexData(datasets, labelInput, label)

    indexedSets.map(this.assembleFeatures(_, label, features))
  }

  /**
    * Preprocess the training and testing datasets by performing: label string indexing,
    * features assembling and features scaling.
    *
    * @param datasets List of datasets
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def preprocessAndScaleData(datasets: Array[DataFrame],
                             label: String, features: String,
                             assembledFeatures: String = "assembledFeatures"): Array[DataFrame] = {

    val preprocessedDatasets = this.preprocessData(datasets, label, assembledFeatures)
    val scaledDatasets = preprocessedDatasets.map(this.scaleFeatures(_, assembledFeatures, features))

    scaledDatasets
  }

  /**
    * Preprocess the training and testing datasets by performing: label string indexing,
    * features assembling, features min-max scaling and features discretization.
    *
    * @param datasets List of datasets
    * @param label    Label output column
    * @param features Features output column
    * @return Array of preprocessed train and test DataFrames
    */
  def preprocessAndDiscretize(datasets: Array[DataFrame],
                              label: String,
                              features: String): Array[DataFrame] = {

    val train = datasets(0)
    val labelInput = train.columns.last

    val indexedDatasets = this.indexData(datasets, labelInput, label)
    val discretizedDatasets = this.discretizeFeatures(indexedDatasets, label)

    discretizedDatasets.map(this.assembleFeatures(_, label, features))
  }

  private def indexData(datasets: Array[DataFrame],
                        inputCol: String,
                        outputCol: String): Array[DataFrame] = {
    val train = datasets(0)

    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .fit(train)

    datasets.map(indexer.transform(_).drop(inputCol))
  }

  def discretizeFeatures(datasets: Array[DataFrame],
                         label: String): Array[DataFrame] = {

    val train = datasets(0)
    val inputFeatures = train.columns.filter(i => !i.equals(label))
    val discretizedFeatures = inputFeatures.map(feature => "discrete_" + feature)

    val discretizer = new QuantileDiscretizer()
      .setInputCols(inputFeatures)
      .setOutputCols(discretizedFeatures)
      .setNumBuckets(Discretize.getNumberOfBeans)
      .fit(train)

    datasets.map(discretizer.transform(_).drop(inputFeatures: _*))
  }

  private def assembleFeatures(data: DataFrame,
                               label: String,
                               assembledFeatures: String): DataFrame = {
    val featuresInput = data.columns.filter(i => !i.equals(label))
    val assembledData = new VectorAssembler()
      .setInputCols(featuresInput)
      .setOutputCol(assembledFeatures)
      .transform(data)

    assembledData.drop(featuresInput: _*)
  }

  private def scaleFeatures(data: DataFrame,
                            inputFeatures: String,
                            outputFeatures: String): DataFrame = {
    val scaler = new MinMaxScaler()
      .setMin(Scale.getMinScaler)
      .setMax(Scale.getMaxScaler)
      .setInputCol(inputFeatures)
      .setOutputCol(outputFeatures)
      .fit(data)

    scaler.transform(data).drop(inputFeatures)
  }

  private def standardizeFeatures(datasets: Array[DataFrame],
                                  label: String,
                                  featuresOutput: String,
                                  assembledFeatures: String = "assembledFeatures"): Array[DataFrame] = {

    val assembledDatasets = datasets.map(this.assembleFeatures(_, label, assembledFeatures))

    val assembledTrain = assembledDatasets(0)
    val scaler = new StandardScaler()
      .setInputCol(assembledFeatures)
      .setOutputCol(featuresOutput)
      .setWithMean(true)
      .setWithStd(true)
      .fit(assembledTrain)

    assembledDatasets.map(scaler.transform(_).drop(assembledFeatures))
  }
}
