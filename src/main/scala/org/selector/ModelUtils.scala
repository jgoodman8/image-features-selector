package org.selector

import org.apache.spark.ml.classification.LogisticRegression

class ModelUtils {

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
}
