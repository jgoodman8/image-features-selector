package ifs.strategies

import ifs.services.ModelService
import ifs.Constants.Classifiers.{DECISION_TREE, LOGISTIC_REGRESSION, MLP, NAIVE_BAYES, RANDOM_FOREST, SVM}
import org.apache.spark.ml.Model
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.DataFrame

object FitStrategy {

  def fit_with_strategy(data: DataFrame, label: String, features: String, method: String): Model[_] with MLWritable = {
    method match {
      case LOGISTIC_REGRESSION => ModelService.fitWithLogisticRegression(data, label, features)
      case RANDOM_FOREST => ModelService.fitWithRandomForest(data, label, features)
      case DECISION_TREE => ModelService.fitWithDecisionTree(data, label, features)
      case MLP => ModelService.fitWithMLP(data, label, features)
      case NAIVE_BAYES => ModelService.fitWithNaiveBayes(data, label, features)
      case SVM => ModelService.fitWithSVM(data, label, features)
      case _ => throw new NoSuchMethodException("The classifier method is not implemented")
    }
  }
}
