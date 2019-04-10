package ifs.jobs

import ifs.Constants
import ifs.services.DataService
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.{ChiSqSelector, InfoThSelector}
import org.apache.spark.sql.{DataFrame, SparkSession}

object FeatureSelectionPipeline extends App with Logging {

  private def selectWithChiSq(sparkSession: SparkSession, train: DataFrame, test: DataFrame, features: String,
                              label: String, selectedFeatures: String, numTopFeatures: Int = 10): Array[DataFrame] = {

    val selectorModel = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    val selectedTrain = selectorModel
      .transform(train)
      .drop(features)
      .select(selectedFeatures, label)

    val selectedTest = selectorModel
      .transform(test)
      .drop(features)
      .select(selectedFeatures, label)

    Array(
      DataService.extractDenseRows(selectedTrain, selectedFeatures, label),
      DataService.extractDenseRows(selectedTest, selectedFeatures, label)
      )
  }

  private def selectWithMRMR(sparkSession: SparkSession, train: DataFrame, test: DataFrame,
                             features: String, label: String, selectedFeatures: String,
                             numTopFeatures: Int = 10): Array[DataFrame] = {

    val selector = new InfoThSelector()
      .setSelectCriterion("mrmr")
      .setNPartitions(100)
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)
      .fit(train)

    val selectedTrain = selector.transform(train).drop(features)
    val selectedTest = selector.transform(test).drop(features)

    Array(
      DataService.extractDenseRows(selectedTrain, selectedFeatures, label),
      DataService.extractDenseRows(selectedTest, selectedFeatures, label)
      )
  }

  def select(session: SparkSession, train: DataFrame, test: DataFrame, features: String, labels: String,
             selectedFeatures: String, method: String, numFeatures: Int = 2): Array[DataFrame] = {
    method match {
      case Constants.CHI_SQ => selectWithChiSq(session, train, test, features, labels, selectedFeatures, numFeatures)
      case Constants.MRMR => selectWithMRMR(session, train, test, features, labels, selectedFeatures, numFeatures)
    }
  }

  def preprocess(train: DataFrame, test: DataFrame, label: String, features: String,
                 method: String): Array[DataFrame] = {
    method match {
      case Constants.CHI_SQ => DataService.preprocessData(train, test, label, features)
      case Constants.MRMR => DataService.preprocessAndDiscretize(train, test, label, features)
    }
  }

  def run(session: SparkSession, trainFile: String, testFile: String, outputPath: String, method: String,
          numFeatures: Int = 100, features: String = "features", label: String = "output_label",
          selectedFeatures: String = "selected"): Unit = {

    val train: DataFrame = DataService.getDataFromFile(session, trainFile)
    val test: DataFrame = DataService.getDataFromFile(session, testFile)

    val Array(preprocessedTrain, preprocessedTest) = this.preprocess(train, test, label, features, method)

    val Array(selectedTrain, selectedTest) = this
      .select(session, preprocessedTrain, preprocessedTest, features, label, selectedFeatures, method, numFeatures)

    DataService.saveData(selectedTrain, fileDir = f"$outputPath%s/train", label, selectedFeatures)
    DataService.saveData(selectedTest, fileDir = f"$outputPath%s/test", label, selectedFeatures)
  }

  val Array(appName: String,
            trainFile: String,
            testFile: String,
            outputPath: String,
            method: String,
            numFeatures: String) = args

  val sparkSession: SparkSession = SparkSession.builder()
    .appName(name = f"$appName%s_$method%s")
    .getOrCreate()

  this.run(sparkSession, trainFile, testFile, outputPath, method, numFeatures.toInt)

  sparkSession.stop()
}
