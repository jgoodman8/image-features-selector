package org.ifs

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.sql._
import org.apache.spark.sql.functions.rand
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
    val testWithFeatures = extractFeaturesByMethod(train, FeatureExtraction.InceptionV3)

    val chiSqSelector: ChiSqSelector = FeatureSelection.getChiSqSelector() // Features Selection
    val trainWithSelectedFeatures: DataFrame = chiSqSelector.fit(trainWithFeatures).transform(trainWithFeatures)

    val logisticRegression: LogisticRegression = ModelUtils.getLogisticRegression() // Model train
    val logisticRegressionModel: LogisticRegressionModel = logisticRegression.fit(trainWithSelectedFeatures)

    val multiClassEvaluator = new MulticlassClassificationEvaluator() // Model training error
      .setMetricName("accuracy")

    val validationPrediction = logisticRegressionModel
      .transform(trainWithFeatures.orderBy(rand()).limit(20))
      .select("prediction", "label")

    print("Training set accuracy = " + multiClassEvaluator.evaluate(validationPrediction).toString)

    // TODO: Select same features in test set as in training set

    val testPrediction = logisticRegressionModel // Model test
      .transform(testWithFeatures)
      .select("prediction", "label")

    print("Test set accuracy = " + multiClassEvaluator.evaluate(testPrediction).toString)
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
