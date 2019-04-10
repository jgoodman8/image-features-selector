package ifs.services

import org.apache.spark.ml.Model
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest, OneVsRestModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object ClassificationService {

  def fitWithLogisticRegression(data: DataFrame, label: String, features: String): OneVsRestModel = {

    val logisticRegression = new LogisticRegression()
      .setMaxIter(ConfigurationService.Model.getMaxIter)
      .setElasticNetParam(ConfigurationService.Model.getElasticNetParam)
      .setRegParam(ConfigurationService.Model.getRegParam)
      .setFeaturesCol(features)
      .setLabelCol(label)

    new OneVsRest()
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setClassifier(logisticRegression)
      .fit(data)
  }

  def fitWithRandomForest(data: DataFrame, label: String, features: String): RandomForestClassificationModel = {
    new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
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

  def saveMetrics(session: SparkSession, metricNames: Array[String], metricValues: Array[Double],
                  outputFolder: String): Unit = {
    import session.implicits._

    metricValues.toSeq.toDF(metricNames: _*)
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(outputFolder + "/" + System.currentTimeMillis.toString)
  }
}
