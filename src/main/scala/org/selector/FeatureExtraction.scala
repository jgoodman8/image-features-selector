package org.selector

import com.databricks.sparkdl.DeepImageFeaturizer

object FeatureExtraction {

  val InceptionV3: String = "InceptionV3"

  def getDeepImageFeaturizer(model: String,
                             inputColumn: String = "image",
                             outputColumn: String = "features"): DeepImageFeaturizer = {
    new DeepImageFeaturizer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .setModelName(model)
  }
}
