package ifs.services

import breeze.linalg.sum
import ifs.services.ConfigurationService.{Model, Session}
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.reflect.ClassTag

object ClassificationService {

  def getEvaluator(label: String, metric: String = Model.getMetrics(0)): Evaluator = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setMetricName(metric)
  }

  def crossValidate[T](data: DataFrame, label: String, features: String, classifier: Estimator[_],
                       params: Array[ParamMap])(implicit tag: ClassTag[T]): T = {

    val crossValidator = new CrossValidator()
      .setEstimator(classifier)
      .setEvaluator(this.getEvaluator(label))
      .setNumFolds(Model.getNumFolds)

    if (Model.isGridSearchActivated && params != null) {
      crossValidator.setEstimatorParamMaps(params)
    } else {
      crossValidator.setEstimatorParamMaps(new ParamGridBuilder().build())
    }

    crossValidator
      .fit(data).bestModel match {
      case model: T => model
      case _ => throw new UnsupportedOperationException("Unexpected model type")
    }
  }

  def evaluate(model: Model[_], data: DataFrame, labelCol: String, metricNames: Array[String],
               predictionCol: String = "prediction"): Array[Double] = {

    val predictions = model.transform(data).select(labelCol, predictionCol)

    metricNames.map(metricName => {
      new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol(predictionCol)
        .setMetricName(metricName)
        .evaluate(predictions)
    })


  }

  def getTopAccuracy(model: Model[_], data: DataFrame, label: String, probability: String = "probability"): Double = {

    val predictions: Array[Row] = model.transform(data).select(label, probability).collect()
    val scores: Array[Int] = predictions.map(this.getInstanceScore(_, label, probability))

    val topAccuracy: Double = sum(scores) / scores.length.toDouble
    topAccuracy
  }

  private def getInstanceScore(row: Row, labelCol: String, probabilityCol: String): Int = {
    val label = row.getAs[Double](labelCol)
    val probabilities: Array[Double] = row.getAs[DenseVector](probabilityCol).toArray

    if (this.getTopLabels(probabilities).contains(label)) 1 else 0
  }

  private def getTopLabels(probabilities: Array[Double]): Array[Double] = {
    val N = Session.getNumTopAccuracy
    val topLabels = probabilities.zipWithIndex.sortBy(_._1).takeRight(N).map(_._2.toDouble)

    topLabels
  }

  def saveMetrics(session: SparkSession, names: Array[String], values: Array[Double], outputFolder: String): Unit = {
    import session.implicits._

    val valuesColumn = names.toList.toDF(colNames = "metric")
      .withColumn(colName = "rowId1", monotonically_increasing_id())
    val namesColumn = values.toList.toDF(colNames = "value")
      .withColumn(colName = "rowId2", monotonically_increasing_id())

    valuesColumn
      .join(namesColumn, valuesColumn("rowId1") === namesColumn("rowId2"))
      .select("metric", "value")
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(outputFolder + System.currentTimeMillis.toString)
  }
}
