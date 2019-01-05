package jobs

import ifs.jobs.ChiSqImageFeatureSelection
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class ChiSqImageFeatureSelectionTest extends FlatSpec with Matchers with BeforeAndAfterAll {

  var sparkSession: SparkSession = _

  override def beforeAll(): Unit = {
    sparkSession = SparkSession.builder()
      .appName("Test Session")
      .master("local")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    sparkSession.stop()
  }

  it should "load the features from a correct csv route" in {
    val csvRoute = "../datasets/inception_v3.csv"

    val features: DataFrame = ChiSqImageFeatureSelection.getDataFromFile(csvRoute, sparkSession)

    assert(features.count() > 0)
  }
}