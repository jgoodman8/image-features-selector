package ifs.utils

import java.nio.file.{Files, Paths}

import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel}
import org.apache.spark.sql.DataFrame

object FeatureSelection {

  val chiSqRoute: String = "hdfs://namenode/chiSqModel"

  def getChiSqSelector(numberOfFeatures: Int = 0,
                       featuresColumn: String = "features",
                       labelColumn: String = "label",
                       selectedFeaturesColumn: String = "selectedFeatures"): ChiSqSelector = {

    val selector = new ChiSqSelector()
      .setFeaturesCol(featuresColumn)
      .setLabelCol(labelColumn)
      .setOutputCol(selectedFeaturesColumn)

    if (numberOfFeatures.>(0)) selector.setNumTopFeatures(numberOfFeatures)

    selector
  }

  /**
    * Creates a model of feature selection by using the ChiSqSelector and stores or loads the model when needed
    *
    * @param data DataFrame with label and features columns
    * @return
    */
  def buildSqModel(data: DataFrame): ChiSqSelectorModel = {
    val chiSqSelectorModel: ChiSqSelectorModel = if (!Files.exists(Paths.get(chiSqRoute))) {
      val model = FeatureSelection.getChiSqSelector().fit(data)
      model.save(chiSqRoute)

      model
    } else {
      ChiSqSelectorModel.load(chiSqRoute)
    }

    chiSqSelectorModel
  }
}
