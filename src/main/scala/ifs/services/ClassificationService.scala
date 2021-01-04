package ifs.services

import breeze.linalg.sum
import ifs.services.ConfigurationService.{Model, Session}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.{MLP, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit, ValidationSplit}
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

  def validate[T](data: DataFrame, validationSet: DataFrame, label: String, classifier: Estimator[_],
                  params: Array[ParamMap] = null)(implicit tag: ClassTag[T]): T = {

    val validator = new ValidationSplit()
      .setEstimator(classifier)
      .setEvaluator(this.getEvaluator(label))
      .setValidationSet(validationSet)

    if (Model.isGridSearchActivated && params != null) {
      validator.setEstimatorParamMaps(params)
    } else {
      validator.setEstimatorParamMaps(new ParamGridBuilder().build())
    }

    validator
      .fit(data).bestModel match {
      case model: T => model
      case _ => throw new UnsupportedOperationException("Unexpected model type")
    }
  }

  def crossValidate[T](data: DataFrame, label: String, classifier: Estimator[_],
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

  def getTopAccuracyN(model: Model[_], data: DataFrame, label: String, probability: String = "probability"): Double = {

    val predictions: Array[Row] = this.getPredictions(model, data, label, probability)
    val scores: Array[Int] = predictions.map(this.getInstanceScore(_, label, probability))

    val topAccuracy: Double = sum(scores) / scores.length.toDouble
    topAccuracy
  }

  private def predictMLP(mlp: MultilayerPerceptronClassificationModel, data: DataFrame, labelCol: String,
                         probabilitiesCol: String, featuresCol: String = "features"): DataFrame = {

    val session = data.sparkSession
    val broadcastModel: Broadcast[MLP] = session.sparkContext.broadcast(new MLP(mlp.uid, mlp.layers, mlp.weights))

    import session.implicits._

    val predictions = data
      .rdd.map((row: Row) => {
      val label = row.getAs[Double](labelCol)
      val features: Vector = row.getAs[Vector](featuresCol)
      val probabilities: DenseVector = broadcastModel.value.customPredict(features).toDense

      (probabilities, label)
    })

    predictions.toDF(probabilitiesCol, labelCol)
  }

  private def getPredictions(model: Model[_], data: DataFrame, label: String, probability: String): Array[Row] = {
    val predictions = model match {
      case model: MultilayerPerceptronClassificationModel => this.predictMLP(model, data, label, probability)
      case model: Model[_] => model.transform(data)
    }

    predictions.select(label, probability).collect()
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
      .coalesce(1)
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(outputFolder + "/" + System.currentTimeMillis.toString)
  }
}
