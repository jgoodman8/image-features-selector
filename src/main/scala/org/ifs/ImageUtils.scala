package org.ifs

import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.ml.image.ImageSchema._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{lit, monotonically_increasing_id}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

class ImageUtils(sparkContext: SparkContext) {

  private val sqlContext = new SQLContext(sparkContext)
  private val trainFolderName: String = "train"
  private val testFolderName: String = "val"

  def loadTrainData(basePath: String): DataFrame = {
    val trainFolder = new File(basePath.concat("/": String).concat(trainFolderName))

    val dataByLabel: List[DataFrame] = trainFolder
      .listFiles
      .map(createImageDFWithLabel)
      .toList

    createImageDataFrame(dataByLabel)
  }

  def loadTestData(basePath: String): DataFrame = {
    val testFolder = new File(basePath.concat("/": String).concat(testFolderName))

    var testLabels = getTestLabels(testFolder)
    var testImages: DataFrame = readImages(this.findImagesPath(testFolder))

    testLabels = testLabels.withColumn("labelsIds", monotonically_increasing_id())
    testImages = testImages.withColumn("imagesIds", monotonically_increasing_id())

    val test = testLabels
      .as("labels")
      .join(testImages.as("images"), testLabels("labelsIds") === testImages("imagesIds"), "inner")
      .select("labels.label", "images.image")
      .toDF("label", "image")

    test
  }

  private def getTestLabels(testFolder: File): DataFrame = {
    val testAnnotations: RDD[Row] = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("delimiter", "\t")
      .option("header", "false")
      .load(testFolder.getAbsolutePath.concat("/val_annotations.txt"))
      .toDF("fileName", "label", "dim1", "dim2", "dim3", "dim4")
      .select("label")
      .rdd
      .map(row => {
        Row(row.getString(0).substring(1: Int).toInt)
      })

    val schema = new StructType().add(StructField("label", IntegerType))

    sqlContext.createDataFrame(testAnnotations, schema)
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
