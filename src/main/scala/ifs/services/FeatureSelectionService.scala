package ifs.services

import org.apache.spark.ml.Model
import org.apache.spark.ml.feature.{ChiSqSelector, InfoThSelector, ReliefFRSelector}
import org.apache.spark.sql.DataFrame

object FeatureSelectionService {

  def selectWithChiSq(train: DataFrame, test: DataFrame, features: String, label: String, selectedFeatures: String,
                      numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    this.transform(selector, train, test, features, label, selectedFeatures)
  }

  def selectWithInfoTheoretic(train: DataFrame, test: DataFrame, features: String, label: String,
                              selectedFeatures: String, method: String, numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new InfoThSelector()
      .setSelectCriterion(method)
      .setNPartitions(100)
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    this.transform(selector, train, test, features, label, selectedFeatures)
  }

  def selectWithRelief(train: DataFrame, test: DataFrame, features: String, label: String, selectedFeatures: String,
                       numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new ReliefFRSelector()
      .setNumTopFeatures(numTopFeatures)
      .setEstimationRatio(0.1)
      .setNumNeighbors(5) // k-NN used in RELIEF
      .setDiscreteData(true)
      .setInputCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

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
