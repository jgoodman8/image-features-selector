package ifs.services

import ifs.services.ConfigurationService.Model
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame

import scala.reflect.ClassTag

object ModelService {

  def fitWithNaiveBayes(data: DataFrame, label: String, features: String): NaiveBayesModel = {
    new NaiveBayes()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
  }

  def fitWithLogisticRegression(data: DataFrame, validation: DataFrame, label: String, features: String): OneVsRestModel = {

    val logisticRegression = new LogisticRegression()
      .setMaxIter(Model.LogisticRegression.getMaxIter)
      .setElasticNetParam(Model.LogisticRegression.getElasticNetParam)
      .setRegParam(Model.LogisticRegression.getRegParam)
      .setFeaturesCol(features)
      .setLabelCol(label)

    val classifier = new OneVsRest()
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setClassifier(logisticRegression)

    this.fitAndValidate[OneVsRestModel](data, validation, label, classifier, null)
  }

  def fitWithRandomForest(data: DataFrame, validation: DataFrame, label: String, features: String): RandomForestClassificationModel = {
    val classifier = new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setNumTrees(Model.RandomForest.getNumTrees)
      .setSubsamplingRate(Model.RandomForest.getSubsamplingRate)
      .setMaxBins(Model.RandomForest.getMaxBins)
      .setMaxDepth(Model.RandomForest.getMaxDepth)

    this.fitAndValidate[RandomForestClassificationModel](data, validation, label, classifier, null)
  }

  def fitWithDecisionTree(data: DataFrame, validation: DataFrame, label: String, features: String): DecisionTreeClassificationModel = {
    val classifier = new DecisionTreeClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxBins(Model.DecisionTree.getMaxBins)
      .setMaxDepth(Model.DecisionTree.getMaxDepth)

    this.fitAndValidate[DecisionTreeClassificationModel](data, validation, label, classifier, null)
  }

  def fitWithMLP(data: DataFrame, validation: DataFrame, label: String, features: String): MultilayerPerceptronClassificationModel = {

    val classifier = new MultilayerPerceptronClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxIter(Model.MLP.getMaxIter)
      .setBlockSize(Model.MLP.getBlockSize)
      .setLayers(Model.MLP.getLayers)

    val gridParams = this.getMLPGridParams(data, label, features, classifier)

    this.fitAndValidate[MultilayerPerceptronClassificationModel](data, validation, label, classifier, gridParams)
  }

  def fitWithSVM(data: DataFrame, validation: DataFrame, label: String, features: String): OneVsRestModel = {

    val linearSVC = new LinearSVC()
      .setMaxIter(Model.LinearSVC.getMaxIter)
      .setRegParam(Model.LinearSVC.getRegParam)
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setRawPredictionCol("rawScores")

    val classifier: OneVsRest = new OneVsRest()
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setClassifier(linearSVC)
      .setParallelism(Model.LinearSVC.getParallelism)

    this.fitAndValidate[OneVsRestModel](data, validation, label, classifier, null)
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

  private def fitAndValidate[T](train: DataFrame, validation: DataFrame, label: String, classifier: Estimator[_],
                                params: Array[ParamMap] = null)(implicit tag: ClassTag[T]): T = {
    if (validation != null)
      ClassificationService.validate[T](train, validation, label, classifier, params)
    else
      ClassificationService.crossValidate[T](train, label, classifier, params)
  }
}
