package org.ifs

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
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
  private val hadoopFileSystem = FileSystem.get(new Configuration())

  def loadTrainData(basePath: String): DataFrame = {

    val trainPath: Path = new Path(basePath.concat("/": String).concat(trainFolderName))

    val dataByLabel: List[DataFrame] = hadoopFileSystem
      .listStatus(trainPath)
      .map(folder => folder.getPath)
      .map(createImageDFWithLabel)
      .toList

    createImageDataFrame(dataByLabel)
  }

  def loadTestData(basePath: String): DataFrame = {
    val testFolder = new Path(basePath.concat("/": String).concat(testFolderName))

    var testLabels = getTestLabels(testFolder)
    val imagesPath: String = testFolder.toUri.toString.concat("/images": String)
    var testImages: DataFrame = readImages(imagesPath)

    testLabels = testLabels.withColumn("labelsIds", monotonically_increasing_id())
    testImages = testImages.withColumn("imagesIds", monotonically_increasing_id())

    val test = testLabels
      .as("labels")
      .join(testImages.as("images"), testLabels("labelsIds") === testImages("imagesIds"), "inner")
      .select("labels.label", "images.image")
      .toDF("label", "image")

    test
  }

  private def getTestLabels(testFolder: Path): DataFrame = {
    val testAnnotations: RDD[Row] = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("delimiter", "\t")
      .option("header", "false")
      .load(testFolder.toUri.toString.concat("/val_annotations.txt"))
      .toDF("fileName", "label", "dim1", "dim2", "dim3", "dim4")
      .select("label")
      .rdd
      .map(row => Row(row.getString(0).substring(1: Int).toInt))

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

  private def createImageDFWithLabel(folder: Path): DataFrame = {
    val label: Int = folder.getName.substring(1: Int).toInt
    val imagesPath: String = folder.asInstanceOf[Path].toUri.toString + "/images"

    readImages(imagesPath)
      .withColumn("label": String, lit(label))
  }
}
