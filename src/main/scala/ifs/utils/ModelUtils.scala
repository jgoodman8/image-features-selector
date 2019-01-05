package ifs.utils

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.sql.DataFrame

object ModelUtils {

  def getLogisticRegression(input: String = "features",
                            label: String = "label",
                            maxIterations: Int = 20,
                            regressionParam: Double = 0.05,
                            elasticNetParam: Double = 0.3): LogisticRegression = {
    new LogisticRegression()
      .setMaxIter(maxIterations)
      .setRegParam(regressionParam)
      .setElasticNetParam(elasticNetParam)
      .setFeaturesCol(input)
      .setLabelCol(label)
  }

  def buildStringIndexerModel(data: DataFrame) = {
    new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
  }

  /**
    * Builds a logistic regression model with the given train data
    *
    * @param data Train DataFrame with label and features column
    * @return
    */
  def trainLogisticRegression(data: DataFrame): LogisticRegressionModel = {
    val logisticRegression: LogisticRegression = ModelUtils.getLogisticRegression()

    logisticRegression.fit(data)
  }

  /**
    * Builds a tree decision tree classification model with the given train data
    *
    * @param data Train DataFrame with label and features column
    * @return
    */
  def trainDecisionTreeClassifier(data: DataFrame): PipelineModel = {
    val indexedData: StringIndexerModel = buildStringIndexerModel(data)

    val decisionTree: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    val pipeline = new Pipeline().setStages(Array(indexedData, decisionTree))

    pipeline.fit(data)
  }
}
