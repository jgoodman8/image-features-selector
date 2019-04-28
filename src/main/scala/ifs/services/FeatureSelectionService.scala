package ifs.services

import ifs.services.ConfigurationService.FeatureSelection
import org.apache.spark.ml.Model
import org.apache.spark.ml.feature.{ChiSqSelector, InfoThSelector, PCA, ReliefFRSelector}
import org.apache.spark.sql.DataFrame

object FeatureSelectionService {

  def selectWithChiSq(train: DataFrame, test: DataFrame, features: String, label: String, selectedFeatures: String,
                      numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train.union(test))

    this.transform(selector, train, test, features, label, selectedFeatures)
  }

  def selectWithInfoTheoretic(train: DataFrame, test: DataFrame, features: String, label: String,
                              selectedFeatures: String, method: String, numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new InfoThSelector()
      .setSelectCriterion(method)
      .setNPartitions(FeatureSelection.InfoTheoretic.getNumberOfPartitions)
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train.union(test))

    this.transform(selector, train, test, features, label, selectedFeatures)
  }

  def selectWithRelief(train: DataFrame, test: DataFrame, features: String, label: String, selectedFeatures: String,
                       numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new ReliefFRSelector()
      .setNumTopFeatures(numTopFeatures)
      .setEstimationRatio(FeatureSelection.Relief.getEstimationRatio)
      .setNumNeighbors(FeatureSelection.Relief.getNumberOfNeighbors) // k-NN used in RELIEF
      .setDiscreteData(FeatureSelection.Relief.isDiscreteData)
      .setInputCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train.union(test))

    this.transform(selector, train, test, features, label, selectedFeatures)
  }

  def selectWithPCA(train: DataFrame, test: DataFrame, features: String, label: String, selectedFeatures: String,
                    numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new PCA()
      .setInputCol(features)
      .setOutputCol(selectedFeatures)
      .setK(numTopFeatures)
      .fit(train.union(test))

    this.transform(selector, train, test, features, label, selectedFeatures)
  }

  private def transform(selector: Model[_], train: DataFrame, test: DataFrame, features: String, label: String,
                        selectedFeatures: String): Array[DataFrame] = {

    val selectedTrain = selector.transform(train).drop(features).select(selectedFeatures, label)
    val selectedTest = selector.transform(test).drop(features).select(selectedFeatures, label)

    Array(
      DataService.extractDenseRows(selectedTrain, selectedFeatures, label),
      DataService.extractDenseRows(selectedTest, selectedFeatures, label)
    )
  }
}
