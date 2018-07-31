package org.selector

import java.io.File

import org.apache.spark.ml.image.ImageSchema._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit

class ImageUtils {

  def loadTrainData(basePath: String): DataFrame = {
    this.loadData(new File(basePath.concat("/train": String)))
  }

  def loadTestData(basePath: String): DataFrame = {
    this.loadData(new File(basePath.concat("/test": String)))
  }

  private def loadData(folder: File): DataFrame = {
    val dataByLabel: List[DataFrame] = folder
      .listFiles
      .map(createImageDFWithLabel)
      .toList

    createImageDataFrame(dataByLabel)
  }

  private def createImageDataFrame(dataFrameList: List[DataFrame]): DataFrame = {
    if (dataFrameList.nonEmpty) {
      var data: DataFrame = dataFrameList.head

      for (labelIndex <- 1 until dataFrameList.length) {
        data = data.union(dataFrameList(labelIndex))
      }

      return data
    }

    throw new Exception("DataFrame List of each image type is empty")
  }

  private def findImagesPath(folder: File): String = {
    folder.getAbsolutePath + "/images"
  }

  private def createImageDFWithLabel(folder: File): DataFrame = {
    val label: Int = folder.getName.substring(1: Int).toInt

    readImages(this.findImagesPath(folder))
      .withColumn("label": String, lit(label))
  }
}
