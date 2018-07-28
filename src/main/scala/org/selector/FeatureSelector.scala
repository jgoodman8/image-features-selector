package org.selector

import java.nio.file.Paths

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.SparkConf
import org.apache.spark.ml.image.ImageSchema._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql._

object FeatureSelector {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("ImageFeatureSelector").setMaster("local")
    //val sparkContext = new SparkContext(sparkConf)

    val testImage = "images"
    //val singleClassImages = "../zeppelin-spark-standalone-cluster/notebooks/tiny-imagenet-200/train/n01443537/images"

    val train_images = readImages(testImage)

    train_images.foreach { rrow =>
      val row = rrow.getAs[Row](0)
      print(row)
      val imageData = getData(row)
      val height = getHeight(row)
      val width = getWidth(row)

      println(s"${height}x${width}")
    }

    train_images.show()

    val featuresExtractor: DeepImageFeaturizer = new DeepImageFeaturizer().setModelName("InceptionV3")
    val pipeline: Pipeline = new Pipeline().setStages(Array(featuresExtractor))

    val model = pipeline.fit(train_images)
  }
}
