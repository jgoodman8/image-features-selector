package jobs

import ifs.jobs.ChiSqImageFeatureSelection
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.util.Random

class ChiSqImageFeatureSelectionTest extends FlatSpec with Matchers with BeforeAndAfterAll {

  val labels = "label"
  val features = "features"
  val selectedFeatures = "features_selected"
  var sparkSession: SparkSession = _
  val csvRoute = "../datasets/inception_v3.csv"

  override def beforeAll(): Unit = {
    sparkSession = SparkSession.builder()
      .appName("Test Session")
      .master("local")
      .getOrCreate()
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
  }

  "selectFeatures" should "select the best features" in {
    val data: DataFrame = ChiSqImageFeatureSelection.getDataFromFile(csvRoute, sparkSession, features, labels)

    val selectedData: DataFrame = ChiSqImageFeatureSelection
      .selectFeatures(data, sparkSession, features, labels, selectedFeatures)

    assert(selectedData.count() > 0)
    assert(selectedData.columns.contains(selectedFeatures))
  }

  "runPipeline" should "do something xD" in {
    ChiSqImageFeatureSelection.runPipeline(sparkSession, csvRoute)
  }
}
