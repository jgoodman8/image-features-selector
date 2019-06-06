package org.apache.spark.ml.classification

import org.apache.spark.annotation.Since
import org.apache.spark.ml.linalg.Vector


@Since("1.5.0")
class MLP(@Since("1.5.0") override val uid: String,
          @Since("1.5.0") override val layers: Array[Int],
          @Since("2.0.0") override val weights: Vector)
  extends MultilayerPerceptronClassificationModel(uid = uid, layers = layers, weights = weights) {

  def customPredict(features: Vector): Vector = this.mlpModel.predict(features)
}
