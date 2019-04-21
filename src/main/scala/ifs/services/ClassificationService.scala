package ifs.services

import ifs.services.ConfigurationService.Model
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
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

    val classifier = new MultilayerPerceptronClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxIter(Model.MLP.getMaxIter)
      .setBlockSize(Model.MLP.getBlockSize)

    val gridParameters = this.getMLPGridParams(data, label, features, classifier)

    this.crossValidate[MultilayerPerceptronClassificationModel](data, label, features, classifier, gridParameters)
  }

  def getMLPGridParams(data: DataFrame, label: String, features: String,
                       classifier: MultilayerPerceptronClassifier): Array[ParamMap] = {

    val inputSize = DataService.getNumberOfFeatures(data, features)
    val outputSize = DataService.getNumberOfLabels(data, label)

    val gridLayers: Array[Array[Int]] = Array(
      Array(inputSize, inputSize./(2).floor.toInt, outputSize),
      Array(inputSize, inputSize.*(2).floor.toInt, outputSize),
      Array(inputSize, outputSize./(2).floor.toInt, outputSize),
      Array(inputSize, outputSize.*(2).floor.toInt, outputSize),
      Array(inputSize, inputSize.*(1.5).floor.toInt, inputSize.*(2.5).floor.toInt, outputSize),
      Array(inputSize, outputSize.*(1.5).floor.toInt, outputSize.*(2.5).floor.toInt, outputSize)
    )

    new ParamGridBuilder()
      .addGrid(classifier.layers, gridLayers)
      .build()
  }

  def getEvaluator(label: String, metric: String = Model.getMetrics(0)): Evaluator = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setMetricName(metric)
  }

  def fitWithNaiveBayes(data: DataFrame, label: String, features: String): NaiveBayesModel = {
    new NaiveBayes()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
  }

  def crossValidate[T](data: DataFrame, label: String, features: String, classifier: Estimator[_],
                       params: Array[ParamMap]): T = {

    val crossValidator = new CrossValidator()
      .setEstimator(classifier)
      .setEvaluator(this.getEvaluator(label))
      .setNumFolds(Model.getNumFolds)

    if (Model.isGridSearchActivated) {
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
