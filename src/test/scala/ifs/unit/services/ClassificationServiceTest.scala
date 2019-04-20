package ifs.unit.services

import ifs.TestUtils
import ifs.services.{ClassificationService, DataService}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ClassificationServiceTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val metricsPath: String = TestUtils.getMetricsOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
    TestUtils.clearDirectory(metricsPath)
  }

  "saveMetrics" should "create a csv file containing the given metrics" in {
    val metricNames = Array("accuracy", "f1")
    val metricValues = Array(1.0, 1.0)

    ClassificationService.saveMetrics(sparkSession, metricNames, metricValues, metricsPath)

    val metricsFile: String = TestUtils.findFileByWildcard(metricsPath)
    val selectedTest: DataFrame = DataService.load(sparkSession, metricsFile)
    assert(selectedTest.columns.length == 2)
    assert(selectedTest.count() == metricNames.length)
  }
}
