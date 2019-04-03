package ifs.jobs

import ifs.{Constants, TestUtils}
import ifs.services.DataService
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FeatureSelectionPipelineTest extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val testData: String = TestUtils.getTestDataRoute
  val outputFile: String = TestUtils.getFeaturesOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
  }

  "runFeatureSelectionPipeline" should "select the best features using the ChiSq method" in {
    val numFeatures = 2
    FeatureSelectionPipeline
      .runFeatureSelectionPipeline(sparkSession, testData, outputFile, Constants.CHI_SQ, numFeatures)

    val selectedData: DataFrame = DataService.getDataFromFile(sparkSession, outputFile)

    assert(selectedData.columns.length == numFeatures + 1)
  }
}
