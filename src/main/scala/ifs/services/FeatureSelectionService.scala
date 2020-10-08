package ifs.services

import ifs.services.ConfigurationService.FeatureSelection
import org.apache.spark.ml.Model
import org.apache.spark.ml.feature.{ChiSqSelector, InfoThSelector, PCA, ReliefFRSelector}
import org.apache.spark.sql.DataFrame

object FeatureSelectionService {

  def selectWithChiSq(datasets: Array[DataFrame],
                      features: String,
                      label: String,
                      selectedFeatures: String,
                      numTopFeatures: Int = 10): Array[DataFrame] = {

    val train = datasets(0)
    val selector = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    this.transform(selector, datasets, features, label, selectedFeatures)
  }

  def selectWithInfoTheoretic(datasets: Array[DataFrame],
                              features: String,
                              label: String,
                              selectedFeatures: String,
                              method: String,
                              numTopFeatures: Int = 10): Array[DataFrame] = {

    val train = datasets(0)
    val selector = new InfoThSelector()
      .setSelectCriterion(method)
      .setNPartitions(FeatureSelection.InfoTheoretic.getNumberOfPartitions)
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    this.transform(selector, datasets, features, label, selectedFeatures)
  }

  def selectWithRelief(datasets: Array[DataFrame],
                       features: String,
                       label: String,
                       selectedFeatures: String,
                       numTopFeatures: Int = 10): Array[DataFrame] = {

    val train = datasets(0)
    val selector = new ReliefFRSelector()
      .setNumTopFeatures(numTopFeatures)
      .setEstimationRatio(FeatureSelection.Relief.getEstimationRatio)
      .setNumNeighbors(FeatureSelection.Relief.getNumberOfNeighbors) // k-NN used in RELIEF
      .setDiscreteData(FeatureSelection.Relief.isDiscreteData)
      .setInputCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    this.transform(selector, datasets, features, label, selectedFeatures)
  }

  def selectWithPCA(datasets: Array[DataFrame],
                    features: String,
                    label: String,
                    selectedFeatures: String,
                    numTopFeatures: Int = 10): Array[DataFrame] = {

    val train = datasets(0)
    val selector = new PCA()
      .setInputCol(features)
      .setOutputCol(selectedFeatures)
      .setK(numTopFeatures)
      .fit(train)

    this.transform(selector, datasets, features, label, selectedFeatures)
  }

  private def transform(selector: Model[_],
                        datasets: Array[DataFrame],
                        features: String,
                        label: String,
                        selectedFeatures: String): Array[DataFrame] = {
    datasets
      .map(selector.transform(_).drop(features).select(selectedFeatures, label))
      .map(DataService.extractDenseRows(_, selectedFeatures, label))
  }
}
