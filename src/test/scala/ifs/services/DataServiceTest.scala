package ifs.services

import ifs.TestUtils
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DataServiceTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val testData: String = TestUtils.getTestDataRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
  }

  "getDataFromFile" should "load the features from a correct csv route" in {
    val data: DataFrame = DataService.getDataFromFile(sparkSession, testData)

    assert(data.count() > 0)
    assert(data.columns.length > 0)

    println(data.first())
  }
}
