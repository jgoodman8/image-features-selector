package ifs.jobs

import ifs.Constants
import ifs.services.DataService
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.{ChiSqSelector, InfoThSelector}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.{DataFrame, SparkSession}

object FeatureSelectionPipeline extends App with Logging {

  private def selectWithChiSq(sparkSession: SparkSession, data: DataFrame, features: String, label: String,
                              selectedFeatures: String, numTopFeatures: Int = 10): DataFrame = {

    val selector = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)

    val selectedData = selector
      .fit(data)
      .transform(data)
      .drop(features)
      .select(selectedFeatures, label)

    DataService.extractDenseRows(selectedData, selectedFeatures, label)
  }

  private def selectWithMRMR(sparkSession: SparkSession, data: DataFrame, features: String, label: String,
                             selectedFeatures: String, numTopFeatures: Int = 10): DataFrame = {

    val selector = new InfoThSelector()
      .setSelectCriterion("mrmr")
      .setNPartitions(100)
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setOutputCol(selectedFeatures)

    val selectedData = selector.fit(data).transform(data).drop(features)

    DataService.extractDenseRows(selectedData, selectedFeatures, label)
  }

  def select(session: SparkSession, data: DataFrame, features: String, labels: String,
             selectedFeatures: String, method: String, numFeatures: Int = 2): DataFrame = {
    method match {
      case Constants.CHI_SQ => selectWithChiSq(session, data, features, labels, selectedFeatures, numFeatures)
      case Constants.MRMR => selectWithMRMR(session, data, features, labels, selectedFeatures, numFeatures)
    }
  }

  def preprocess(data: DataFrame, label: String, features: String, method: String): DataFrame = {
    method match {
      case Constants.CHI_SQ => DataService.preprocessData(data, label, features)
      case Constants.MRMR => DataService.preprocessAndDiscretize(data, label, features)
    }
  }

  def run(session: SparkSession, inputFile: String, outputFile: String, method: String, numFeatures: Int = 100,
          features: String = "features", label: String = "output_label", selected: String = "selected"): Unit = {

    val data = DataService.getDataFromFile(session, inputFile)

    val preprocessedData = this.preprocess(data, label, features, method)

    val selectedData = select(session, preprocessedData, features, label, selected, method, numFeatures)

    DataService.saveData(selectedData, outputFile, label, selected)
  }

  val Array(appName: String, inputFile: String, outputFile: String, method: String, numFeatures: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  this.run(sparkSession, inputFile, outputFile, method, numFeatures.toInt)

  sparkSession.stop()
}
