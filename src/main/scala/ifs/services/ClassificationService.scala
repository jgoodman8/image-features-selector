package ifs.services

import ifs.services.ConfigurationService.Model
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.monotonically_increasing_id

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

  def evaluate(model: Model[_], test: DataFrame, labelCol: String, metricNames: Array[String],
               predictionCol: String = "prediction"): Array[Double] = {

    val predictions = model.transform(test).select(labelCol, predictionCol)

    metricNames.map(metricName => {
      new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol(predictionCol)
        .setMetricName(metricName)
        .evaluate(predictions)
    })
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
