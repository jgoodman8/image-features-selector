package org.ifs

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.sql.DataFrame

object FeatureExtraction {

  val InceptionV3: String = "InceptionV3"
  val ResNet50: String = "ResNet50"

  def getDeepImageFeaturizer(model: String,
                             inputColumn: String = "image",
                             outputColumn: String = "features"): DeepImageFeaturizer = {
    new DeepImageFeaturizer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .setModelName(model)
  }

  /**
    * Extracts a DataFrame of features from the given DataFrame of images
    *
    * @param data   DataFrame of images with labels
    * @param method A FeatureExtraction method
    * @return
    */
  def extractFeaturesByMethod(data: DataFrame, method: String = FeatureExtraction.InceptionV3): DataFrame = {

    method match {
      case FeatureExtraction.InceptionV3 =>
        val featuresExtractor: DeepImageFeaturizer = FeatureExtraction
          .getDeepImageFeaturizer(FeatureExtraction.InceptionV3)

        val trainWithFeatures: DataFrame = featuresExtractor
          .transform(data)
          .select("features", "label")

        trainWithFeatures

      case FeatureExtraction.ResNet50 =>
        val featuresExtractor: DeepImageFeaturizer = FeatureExtraction
          .getDeepImageFeaturizer(ResNet50)

        val trainWithFeatures: DataFrame = featuresExtractor
          .transform(data)
          .select("features", "label")

        trainWithFeatures

      case _ => null
    }
  }
}
