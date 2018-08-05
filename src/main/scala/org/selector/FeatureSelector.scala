package org.selector

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}


object FeatureSelector {

  def main(args: Array[String]): Unit = {
    val Array(basePath: String, master: String) = args

    val sparkConfiguration = new SparkConf()
      .setAppName("ImageFeatureSelector")
      .setMaster(master)
      .set("spark.executor.memory", "1g")
      .set("spark.executor.cores", "4")

    val sparkContext = new SparkContext(sparkConfiguration)

    val Array(train, test) = getAllDataSets(basePath)

    val featuresExtractor: DeepImageFeaturizer = FeatureExtraction
      .getDeepImageFeaturizer(FeatureExtraction.InceptionV3)

    val transformed: DataFrame = featuresExtractor.transform(train)
    val head: Array[Row] = transformed.head(5)

    //    val features = transformed.schema(transformed.schema.fieldIndex("features"))
    //    val meta: Metadata = features.metadata
    //    val mlAttr = meta.getMetadata("ml_attr").getMetadata("attrs")
  }

  private def getAllDataSets(basePath: String): Array[DataFrame] = {
    val imageUtils = new ImageUtils()
    val train = imageUtils.loadTrainData(basePath)
    val test = imageUtils.loadTestData(basePath)

    Array(train, test)
  }
}
