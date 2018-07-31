package org.selector

import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.ml.image.ImageSchema.readImages
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Encoders, Row, SQLContext}

class ImageUtils(sparkContext: SparkContext) {

  private val sqlContext = new SQLContext(sparkContext)

  def loadTrainData(basePath: String): DataFrame = {
    this.loadData(new File(basePath.concat("/train")))
  }

  def loadTestData(basePath: String): DataFrame = {
    this.loadData(new File(basePath.concat("/test")))
  }

  private def loadData(folder: File): DataFrame = {
    val data: Seq[(Row, String)] = folder
      .listFiles
      .flatMap(createImageDFWithLabel) //(_: (Encoders.type [Row], String))
      .toSeq

    this.sqlContext
      .createDataFrame(this.sparkContext.parallelize(data))
      .toDF("image", "label")
  }

  private def findImagesPath(folder: File) = {
    folder.getAbsolutePath + "/images"
  }

  private def createImageDFWithLabel(folder: File): Array[(Row, String)] = {
    val dataFolder = this.findImagesPath(folder)

    readImages(dataFolder)
      .withColumn("label", lit(folder.getName))
      .collect()
      .map(x => (x.getAs[Row](0), x.getString(1)))
    //      .asInstanceOf[Array[(Any, String)]]
  }
}
