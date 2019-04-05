package ifs.unit.jobs

import ifs.jobs.ClassificationPipeline
import ifs.{Constants, TestUtils}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ClassificationPipelineTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val testData: String = TestUtils.getTestDataRoute
  val modelsPath: String = TestUtils.getModelsRoute
  val metricsPath: String = TestUtils.getMetricsOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
  }

  "trainPipeline" should "classify the dataset using a logistic regression model" in {
    val method = Constants.LOGISTIC_REGRESSION
    ClassificationPipeline.run(sparkSession, testData, metricsPath, modelsPath, method)
  }

  it should "classify the dataset using a random forest model" in {
    val method = Constants.RANDOM_FOREST
    ClassificationPipeline.run(sparkSession, testData, metricsPath, modelsPath, method)
  }
}
