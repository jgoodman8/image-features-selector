package org.ifs

import org.apache.spark.ml.classification.LogisticRegression

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
}
