package org.ifs

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}


object IFSResearch {

  def main(args: Array[String]): Unit = {
    val Array(basePath: String, master: String) = args

    val sparkConfiguration = new SparkConf()
      .setMaster(master)
      .setAppName("ImageFeatureSelector")
      .set("spark.executor.memory", "1g")
      .set("spark.executor.cores", "4")

    val sparkContext = new SparkContext(sparkConfiguration)

    val Array(train, test) = loadDataSets(basePath, sparkContext)

    val trainWithFeatures = extractFeaturesByMethod(train, FeatureExtraction.InceptionV3) // Features Extraction

    val chiSqSelector: ChiSqSelector = FeatureSelection.getChiSqSelector() // Features Selection
    val trainWithSelectedFeatures: DataFrame = chiSqSelector.fit(trainWithFeatures).transform(trainWithFeatures)

    val logisticRegression: LogisticRegression = ModelUtils.getLogisticRegression() // Model train
    val logisticRegressionModel: LogisticRegressionModel = logisticRegression.fit(trainWithSelectedFeatures)

    println(s"Coefficients: ${logisticRegressionModel.coefficients} Intercept: ${logisticRegressionModel.intercept}")
  }

  private def loadDataSets(basePath: String, sparkContext: SparkContext): Array[DataFrame] = {
    val imageUtils = new ImageUtils(sparkContext)

    val train = imageUtils.loadTrainData(basePath)
    val test = imageUtils.loadTestData(basePath)

    Array(train, test)
  }

  private def extractFeaturesByMethod(train: DataFrame, method: String): DataFrame = {

    method match {
      case FeatureExtraction.InceptionV3 => {
        val featuresExtractor: DeepImageFeaturizer = FeatureExtraction
          .getDeepImageFeaturizer(FeatureExtraction.InceptionV3)

        val trainWithFeatures: DataFrame = featuresExtractor
          .transform(train)
          .select("features", "label")

        trainWithFeatures
      }
      case _ => null
    }
  }
}
