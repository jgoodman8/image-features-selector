package ifs.utils

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.ChiSqSelectorModel
import org.apache.spark.sql._
import org.apache.spark.sql.functions.rand
import org.apache.spark.{SparkConf, SparkContext}
import ModelUtils.buildStringIndexerModel


object IFSResearch {

  def main(args: Array[String]): Unit = {
    val Array(basePath: String,
              master: String,
              executorMemory: String,
              executorCores: String,
              driverCores: String,
              driverMemory: String) = args

    val sparkConfiguration = new SparkConf()
      .setMaster(master)
      .setAppName("ImageFeatureSelector")
      .set("spark.executor.cores", executorCores)
      .set("spark.executor.memory", executorMemory)
      .set("spark.driver.cores", driverCores)
      .set("spark.driver.memory", driverMemory)

    val sparkContext: SparkContext = new SparkContext(sparkConfiguration)

    runResNet50WithDecisionTree(basePath, sparkContext)
  }

  /**
    * Full pipeline using a pre-trained CCN InceptionV3 for feature extraction, ChiSq for feature selection and
    * Logistic Regression to create to model.
    *
    * @param basePath     Imagenet base folder route
    * @param sparkContext Current Spark context
    */
  private def runInceptionV3WithLogisticRegression(basePath: String, sparkContext: SparkContext): Unit = {
    val Array(train, test) = loadDataSets(basePath, sparkContext)

    val trainWithFeatures = FeatureExtraction.extractFeaturesByMethod(train, FeatureExtraction.InceptionV3)
    val testWithFeatures = FeatureExtraction.extractFeaturesByMethod(test, FeatureExtraction.InceptionV3)

    val chiSqSelectorModel: ChiSqSelectorModel = FeatureSelection.buildSqModel(trainWithFeatures)

    val trainWithSelectedFeatures = chiSqSelectorModel.transform(trainWithFeatures)
    val testWithSelectedFeatures: DataFrame = chiSqSelectorModel.transform(testWithFeatures)

    val logisticRegressionModel: LogisticRegressionModel = ModelUtils.trainLogisticRegression(trainWithSelectedFeatures)

    val multiClassEvaluator = new MulticlassClassificationEvaluator() // Model training error
      .setMetricName("accuracy")

    val validationPrediction = logisticRegressionModel
      .transform(trainWithFeatures.orderBy(rand()).limit(20))
      .select("prediction", "label")

    println("Training set accuracy = " + multiClassEvaluator.evaluate(validationPrediction).toString)

    val testPrediction = logisticRegressionModel // Model test
      .transform(testWithSelectedFeatures)
      .select("prediction", "label")

    println("Test set accuracy = " + multiClassEvaluator.evaluate(testPrediction).toString)
  }

  /**
    * Full pipeline using a pre-trained CCN ResNet50 for feature extraction, ChiSq for feature selection and
    * Tree Decision Classification to create to model.
    *
    * @param basePath     Imagenet base folder route
    * @param sparkContext Current Spark context
    */
  private def runResNet50WithDecisionTree(basePath: String, sparkContext: SparkContext): Unit = {
    val Array(train, test) = loadDataSets(basePath, sparkContext)

    val trainWithFeatures = FeatureExtraction.extractFeaturesByMethod(train, FeatureExtraction.InceptionV3)
    val testWithFeatures = FeatureExtraction.extractFeaturesByMethod(test, FeatureExtraction.InceptionV3)

    val chiSqSelectorModel: ChiSqSelectorModel = FeatureSelection.buildSqModel(trainWithFeatures)

    val trainWithSelectedFeatures = chiSqSelectorModel.transform(trainWithFeatures)
    val testWithSelectedFeatures: DataFrame = chiSqSelectorModel.transform(testWithFeatures)

    val treeDecisionModel: PipelineModel = ModelUtils.trainDecisionTreeClassifier(trainWithSelectedFeatures)

    val multiClassEvaluator = new MulticlassClassificationEvaluator() // Model training error
      .setMetricName("accuracy")

    val validationPrediction = treeDecisionModel
      .transform(trainWithFeatures.orderBy(rand()).limit(20))
      .select("prediction", "label")

    println("Training set accuracy = " + multiClassEvaluator.evaluate(validationPrediction).toString)

    val testIndexed = buildStringIndexerModel(testWithFeatures).transform(testWithFeatures)

    val testPrediction = treeDecisionModel // Model test
      .transform(testIndexed)
      .select("prediction", "label")

    println("Test set accuracy = " + multiClassEvaluator.evaluate(testPrediction).toString)
  }

  /**
    * Loads both training and test sets as Spark DataFrames
    *
    * @param basePath     Imagenet base folder route
    * @param sparkContext Current Spark context
    * @return
    */
  private def loadDataSets(basePath: String, sparkContext: SparkContext): Array[DataFrame] = {
    val imageUtils = new ImageUtils(sparkContext)

    val train = imageUtils.loadTrainData(basePath)
    val test = imageUtils.loadTestData(basePath)

    Array(train, test)
  }
}
