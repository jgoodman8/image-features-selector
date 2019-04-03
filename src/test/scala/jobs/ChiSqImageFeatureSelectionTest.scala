package jobs

import ifs.jobs.ChiSqImageFeatureSelection
import ifs.jobs.ChiSqImageFeatureSelection.{inputFile, metricsPath, modelsPath, sparkSession}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.util.Random

class ChiSqImageFeatureSelectionTest extends FlatSpec with Matchers with BeforeAndAfterAll {

  val features = "features"
  val labels = "output_label"
  val selectedFeatures = "features_selected"
  var sparkSession: SparkSession = _
  val csvRoute = "/home/jgfigueira/datasets/imagenet-features/vgg19.csv"

  override def beforeAll(): Unit = {
    sparkSession = SparkSession.builder()
      .appName("Test Session")
      .master("local[4]")
      .config("spark.driver.memory", "6g")
      .config("spark.executor.memory", "6g")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.shuffle.spill", "false")
      .getOrCreate()

    sparkSession.sparkContext.setCheckpointDir("/tmp/spark/checkpoint")
  }

  override def afterAll(): Unit = {
    sparkSession.stop()
  }

  "toDenseDF" should "transform a DataFrame by adding a label column and a features dense one" in {
    val dataset = sparkSession.createDataFrame(
      Seq((Random.nextDouble(), Random.nextDouble(), Random.nextDouble()))
    ).toDF("x", "y", "z")

    val data = ChiSqImageFeatureSelection.preprocessData(dataset, features, labels)

    assert(data.count() == 1)
    assert(data.columns.length == 5)
    assert(data.columns.contains(features))
    assert(data.columns.contains(labels))
  }

  "getDataFromFile" should "load the features from a correct csv route" in {
    val data: DataFrame = ChiSqImageFeatureSelection.getDataFromFile(csvRoute, sparkSession)

    assert(data.count() > 0)
    assert(data.columns.contains(features))
    assert(data.columns.contains(labels))

    println(data.first())
  }

  "selectFeatures" should "select the best features" in {
    val data: DataFrame = ChiSqImageFeatureSelection.getDataFromFile(csvRoute, sparkSession)

    val selectedData: DataFrame = ChiSqImageFeatureSelection
      .selectFeatures(data, sparkSession, features, labels, selectedFeatures)

    assert(selectedData.count() > 0)
    assert(selectedData.columns.contains(selectedFeatures))
  }

  "trainPipeline" should "work" in {

    val metricsPath = "./output/metrics.csv"
    val modelsPath = "./output/models"
    val inputFile = "./data/data.csv"

    ChiSqImageFeatureSelection.runTrainPipeline(sparkSession, inputFile, metricsPath, modelsPath)
  }
}
