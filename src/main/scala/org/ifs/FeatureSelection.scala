package org.ifs

import org.apache.spark.ml.feature.ChiSqSelector

object FeatureSelection {

  def getChiSqSelector(numberOfFeatures: Int = 0,
                       featuresColumn: String = "features",
                       labelColumn: String = "label",
                       selectedFeaturesColumn: String = "selectedFeatures"): ChiSqSelector = {
    // TODO: Add other selection methods for ChiSq (now using numTopFeatures by default)

    val selector = new ChiSqSelector()
      .setFeaturesCol(featuresColumn)
      .setLabelCol(labelColumn)
      .setOutputCol(selectedFeaturesColumn)

    if (numberOfFeatures.>(0)) selector.setNumTopFeatures(numberOfFeatures)

    selector
  }
}
