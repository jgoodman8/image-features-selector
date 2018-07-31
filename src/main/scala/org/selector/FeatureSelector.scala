package org.selector

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}


object FeatureSelector {

  def main(args: Array[String]): Unit = {
    val sparkConfiguration = new SparkConf()
      .setAppName("ImageFeatureSelector")
      .setMaster("local[4]")
    val sparkContext = new SparkContext(sparkConfiguration)

    val Array(basePath: String) = args

    val imageUtils = new ImageUtils()
    val train: DataFrame = imageUtils
      .loadTrainData(basePath)
    // val test: DataFrame = imageUtils.loadTestData(basePath)

    val featuresExtractor: DeepImageFeaturizer = new DeepImageFeaturizer()
      .setInputCol("image")
      .setOutputCol("features")
      .setModelName("InceptionV3")

    val logisticRegression = new LogisticRegression()
      .setMaxIter(20)
      .setRegParam(0.05)
      .setElasticNetParam(0.3)
      .setLabelCol("label")

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(featuresExtractor, logisticRegression))

    val model = pipeline.fit(train)

    println(model)
  }

}
