package ifs.services

import ifs.services.ConfigurationService.Model
import ifs.services.ClassificationService.crossValidate
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame

object ModelService {

  def fitWithNaiveBayes(data: DataFrame, label: String, features: String): NaiveBayesModel = {
    new NaiveBayes()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .fit(data)
  }

  def fitWithLogisticRegression(data: DataFrame, label: String, features: String): OneVsRestModel = {

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

    crossValidate[OneVsRestModel](data, label, features, classifier, null)
  }

  def fitWithRandomForest(data: DataFrame, label: String, features: String): RandomForestClassificationModel = {
    val classifier = new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setNumTrees(Model.RandomForest.getNumTrees)
      .setSubsamplingRate(Model.RandomForest.getSubsamplingRate)

    if (Model.RandomForest.hasMaxDepth) {
      classifier.setMaxDepth(Model.RandomForest.getMaxDepth)
    }

    crossValidate[RandomForestClassificationModel](data, label, features, classifier, null)
  }

  def fitWithDecisionTree(data: DataFrame, label: String, features: String): DecisionTreeClassificationModel = {
    val classifier = new DecisionTreeClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)

    crossValidate[DecisionTreeClassificationModel](data, label, features, classifier, null)
  }

  def fitWithMLP(data: DataFrame, label: String, features: String): MultilayerPerceptronClassificationModel = {

    val classifier = new MultilayerPerceptronClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxIter(Model.MLP.getMaxIter)
      .setBlockSize(Model.MLP.getBlockSize)

    val gridParameters = this.getMLPGridParams(data, label, features, classifier)

    crossValidate[MultilayerPerceptronClassificationModel](data, label, features, classifier, gridParameters)
  }

  def fitWithSVM(data: DataFrame, label: String, features: String): OneVsRestModel = {

    val linearSVC = new LinearSVC()
      .setMaxIter(Model.LinearSVC.getMaxIter)
      .setRegParam(Model.LinearSVC.getRegParam)
      .setFeaturesCol(features)
      .setLabelCol(label)

    val classifier: OneVsRest = new OneVsRest()
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setClassifier(linearSVC)

    crossValidate[OneVsRestModel](data, label, features, classifier, null)
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
}
