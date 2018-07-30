package org.selector

import java.io.File

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.image.ImageSchema._
import org.apache.spark.sql.functions.lit
import org.apache.spark.{SparkConf, SparkContext}


object FeatureSelector {

  def main(args: Array[String]): Unit = {
    val sparkConfiguration = new SparkConf().setAppName("ImageFeatureSelector").setMaster("local[2]")
    val sparkContext = new SparkContext(sparkConfiguration)

    val trainImagesFolder = new File("../tiny-imagenet-200/train")
    val train = trainImagesFolder.listFiles()
      .map((imageFolder: File) => {
        readImages(imageFolder.getAbsolutePath + "/images")
          .withColumn("label", lit(imageFolder.getName))
      })

    //    val train_images = readImages(testImage)
    //    val myUDf = udf(() => Array("test"))

    //    train_images.withColumn("label", myUDf())
    //    train_images.show()

    val featuresExtractor: DeepImageFeaturizer = new DeepImageFeaturizer()
      .setModelName("InceptionV3")
    val logisticRegression = new LogisticRegression()
      .setMaxIter(20)
      .setRegParam(0.05)
      .setElasticNetParam(0.3)
      .setLabelCol("label")

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(featuresExtractor, logisticRegression))

    val model = pipeline.fit(train)

    //    println(model)
  }

}
