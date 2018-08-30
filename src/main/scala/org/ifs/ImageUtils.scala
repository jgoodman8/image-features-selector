package org.ifs

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.ml.image.ImageSchema._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{lit, monotonically_increasing_id}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs

class ImageUtils(sparkContext: SparkContext) {

  private val sqlContext = new SQLContext(sparkContext)
  private val trainFolderName: String = "train"
  private val testFolderName: String = "val"
  private val hadoopFileSystem = FileSystem.get(new Configuration())

  case class ImageRow(label: Int, image: Mat)

  /**
    * Builds a train set DataFrame with label and image columns, loading train data from a given image base route.
    *
    * @param basePath Imagenet base route
    * @return
    */
  def loadTrainData(basePath: String): DataFrame = {

    val trainPath: Path = new Path(basePath.concat("/": String).concat(trainFolderName))

    val dataByLabel: List[DataFrame] = hadoopFileSystem
      .listStatus(trainPath)
      .map(folder => folder.getPath)
      .map(createImageDFWithLabel)
      .toList

    createImageDataFrame(dataByLabel)
  }

  /**
    * Builds a test set DataFrame with label and image columns, loading validation data from a given image base route.
    *
    * @param basePath Imagenet base route
    * @return
    */
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

  /**
    * Create train data set with OpenCV methods
    * [Not currently in use, check Python version]
    *
    * @param basePath Base imagenet route
    * @return
    */
  def loadTrainDataAsMatrix(basePath: String): DataFrame = {
    val trainPath: Path = new Path(basePath.concat("/": String).concat(trainFolderName))

    val labeledImages: Array[(Int, Path)] = hadoopFileSystem
      .listStatus(trainPath)
      .map(folder => folder.getPath.toUri.toString.concat("/images"))
      .flatMap((imagesFolder: String) => {
        hadoopFileSystem
          .listStatus(new Path(imagesFolder))
          .map(image => image.getPath)
      })
      .map((imagePath: Path) => {
        val label: Int = imagePath.getName.split("_").head.substring(1: Int).toInt

        (label, imagePath)
      })

    val labeledImageMatrix: Array[(Int, Mat)] = labeledImages
      .map(labeledImage => {
        val imageFilename: String = labeledImage._2.toUri.toString
        val imageMatrix: Mat = Imgcodecs.imread(imageFilename)

        (labeledImage._1, imageMatrix)
      })

    val trainRDD = sparkContext
      .parallelize(labeledImageMatrix)
      .map(labeledItem => ImageRow(labeledItem._1, labeledItem._2))

    sqlContext.createDataFrame(trainRDD)
  }

  /**
    * Builds a data set with a column of labels, by reading a tsv file
    *
    * @param testFolder Test path, which contains a TSV file decribing the files names, labels and frames coordinates.
    * @return
    */
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

  /**
    * Builds a DataFrame, from a list of data frames by "flatten" the list
    *
    * @param dataFrameList List of DataFrames (one per image type), each one with a image and a label column.
    * @return
    */
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

  /**
    * Creates a DataFrame with both label and images columns, by reading a given folder and loading the included images
    *
    * @param folder Path that contains images from a single label
    * @return
    */
  private def createImageDFWithLabel(folder: Path): DataFrame = {
    val label: Int = folder.getName.substring(1: Int).toInt
    val imagesPath: String = folder.toUri.toString + "/images"

    readImages(imagesPath)
      .withColumn("label": String, lit(label))
  }
}
