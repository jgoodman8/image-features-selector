package ifs.strategies

import ifs.services.{ClassificationService, ConfigurationService}
import ifs.Constants.Classifiers._
import org.apache.spark.ml.Model
import org.apache.spark.sql.DataFrame

object EvaluationStrategy {
  def evaluate_with_strategy(method: String, model: Model[_], train: DataFrame, test: DataFrame,
                             label: String): (Array[String], Array[Double], Array[Double]) = {

    var metricNames: Array[String] = ConfigurationService.Model.getMetrics
    var trainMetricValues: Array[Double] = ClassificationService.evaluate(model, train, label, metricNames)
    var testMetricValues: Array[Double] = ClassificationService.evaluate(model, test, label, metricNames)


    if (method == NAIVE_BAYES || method == DECISION_TREE || method == RANDOM_FOREST || method == MLP) {
      val trainTopAccuracy = ClassificationService.getTopAccuracyN(model, train, label)
      val testTopAccuracy = ClassificationService.getTopAccuracyN(model, test, label)

      metricNames = metricNames :+ "topNAccuracy"
      trainMetricValues = trainMetricValues :+ trainTopAccuracy
      testMetricValues = testMetricValues :+ testTopAccuracy
    }

    (metricNames, trainMetricValues, testMetricValues)
  }
}
