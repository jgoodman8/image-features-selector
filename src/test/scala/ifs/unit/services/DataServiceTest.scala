package ifs.unit.services

import ifs.TestUtils
import ifs.services.DataService
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DataServiceTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val dataFile: String = TestUtils.getTestDataRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
  }

  "load" should "load the features from a correct csv route" in {
    val data: DataFrame = DataService.load(sparkSession, dataFile)

    assert(data.count() > 0)
    assert(data.columns.length > 0)
  }
}
