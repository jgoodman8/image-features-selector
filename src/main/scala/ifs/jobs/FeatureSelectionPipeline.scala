package ifs.jobs

import ifs.services.DataService
import ifs.Constants
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

object FeatureSelectionPipeline extends App with Logging {

  private def selectWithChiSq(sparkSession: SparkSession, data: DataFrame, features: String, labels: String,
                              selectedFeatures: String, numTopFeatures: Int = 10): DataFrame = {

    val selector = new ChiSqSelector()
      .setNumTopFeatures(numTopFeatures)
      .setFeaturesCol(features)
      .setLabelCol(labels)
      .setOutputCol(selectedFeatures)

    selector
      .fit(data)
      .transform(data)
      .drop(features)
      .select(selectedFeatures, labels)
  }

  private def extractDenseRows(data: DataFrame, features: String, labels: String): DataFrame = {

    val columnsSize = data.first.getAs[DenseVector](features).size + 1

    val vecToSeq = udf((v: DenseVector, label: Double) => v.toArray :+ label)
    val exprs = (0 until columnsSize).map(i => col("_tmp").getItem(i).alias(s"f$i"))

    data.select(vecToSeq(col(features), col(labels)).alias("_tmp")).select(exprs: _*)
  }

  def select(sparkSession: SparkSession, data: DataFrame, features: String, labels: String,
             selectedFeatures: String, method: String, numFeatures: Int = 2): DataFrame = {
    method match {
      case Constants.CHI_SQ => this.selectWithChiSq(sparkSession, data, features, labels, selectedFeatures, numFeatures)
    }
  }

  def runFeatureSelectionPipeline(session: SparkSession, inputFile: String, outputFile: String, method: String,
                                  numFeatures: Int = 100, features: String = "features", label: String = "output_label",
                                  selectedFeatures: String = "selected_features"): Unit = {

    var data = DataService.getDataFromFile(session, inputFile)
    data = DataService.preprocessData(data, features, label)

    val selectedData = this.select(session, data, features, label, selectedFeatures, method, numFeatures)
    this.extractDenseRows(selectedData, selectedFeatures, label)
      .write.mode(SaveMode.Overwrite)
      .csv(outputFile)
  }

  val Array(appName: String, inputFile: String, outputFile: String, method: String, numFeatures: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  this.runFeatureSelectionPipeline(sparkSession, inputFile, outputFile, method, numFeatures.toInt)

  sparkSession.stop()
}
