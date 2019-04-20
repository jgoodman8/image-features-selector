package ifs.services

import ifs.services.ConfigurationService.Model
import org.apache.spark.ml.Model
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object ClassificationService {

  def fitWithLogisticRegression(data: DataFrame, label: String, features: String): OneVsRestModel = {

    val logisticRegression = new LogisticRegression()
      .setMaxIter(Model.LogisticRegression.getMaxIter)
      .setElasticNetParam(Model.LogisticRegression.getElasticNetParam)
      .setRegParam(Model.LogisticRegression.getRegParam)
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

  def fitWithDecisionTree(data: DataFrame, label: String, features: String): DecisionTreeClassificationModel = {
    new DecisionTreeClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
  }

  def fitWithMLP(data: DataFrame, label: String, features: String): MultilayerPerceptronClassificationModel = {
    val inputLayerSize = DataService.getNumberOfFeatures(data, features)
    val outputLayerSize = DataService.getNumberOfLabels(data, label)

    new MultilayerPerceptronClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setLayers(Array(inputLayerSize, outputLayerSize * 2, outputLayerSize))
      .setMaxIter(Model.MLP.getMaxIter)
      .setBlockSize(Model.MLP.getBlockSize)
      .fit(data)
  }

  def fitWithNaiveBayes(data: DataFrame, label: String, features: String): NaiveBayesModel = {
    new NaiveBayes()
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

    val valuesColumn = metricNames.toList.toDF(colNames = "metric")
      .withColumn(colName = "rowId1", monotonically_increasing_id())
    val namesColumn = metricValues.toList.toDF(colNames = "value")
      .withColumn(colName = "rowId2", monotonically_increasing_id())


    valuesColumn
      .join(namesColumn, valuesColumn("rowId1") === namesColumn("rowId2"))
      .select("metric", "value")
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(outputFolder + "/" + System.currentTimeMillis.toString)
  }
}
