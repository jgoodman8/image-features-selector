package jobs

import java.nio.file.{Files, Paths}

import ifs.jobs.ChiSqImageFeatureSelection
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.util.Random

class ChiSqImageFeatureSelectionTest extends FlatSpec with Matchers with BeforeAndAfterAll {

  val features = "features"
  val labels = "output_label"
  val selectedFeatures = "features_selected"
  var sparkSession: SparkSession = _
  val csvRoute = "../datasets/inception_v3.csv"

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

    val data = ChiSqImageFeatureSelection.toDenseDF(dataset, features, labels)

    assert(data.count() == 1)
    assert(data.columns.length == 5)
    assert(data.columns.contains(features))
    assert(data.columns.contains(labels))
  }

  "getDataFromFile" should "load the features from a correct csv route" in {
    val data: DataFrame = ChiSqImageFeatureSelection.getDataFromFile(csvRoute, sparkSession, features, labels)

    assert(data.count() > 0)
    assert(data.columns.contains(features))
    assert(data.columns.contains(labels))

    println(data.first())
  }

  "selectFeatures" should "select the best features" in {
    val data: DataFrame = ChiSqImageFeatureSelection.getDataFromFile(csvRoute, sparkSession, features, labels)

    val selectedData: DataFrame = ChiSqImageFeatureSelection
      .selectFeatures(data, sparkSession, features, labels, selectedFeatures)

    assert(selectedData.count() > 0)
    assert(selectedData.columns.contains(selectedFeatures))
  }

  "runPipeline" should "run the full pipeline without any failure" in {
    ChiSqImageFeatureSelection.runPipeline(sparkSession, csvRoute, "./output/")
  }

  "saveMetrics" should "create a csv File" in {
    val names = Array("a", "b", "c", "d")
    val values = Array(1.0, 2.0, 3.0, 4.0)
    val folder = "./output/" + System.currentTimeMillis().toString + ".csv"

    ChiSqImageFeatureSelection.saveMetrics(sparkSession, names, values, folder)

    Files.exists(Paths.get(folder))
  }
}
