package ifs.validation

import ifs.jobs.{ClassificationPipeline, FeatureSelectionPipeline}
import ifs.{Constants, TestUtils}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ValidationTests extends FlatSpec with Matchers with BeforeAndAfter {

  var sparkSession: SparkSession = _
  val testFile: String = TestUtils.getTestDataRoute
  val outputFile: String = TestUtils.getFeaturesOutputRoute
  val modelsPath: String = TestUtils.getModelsRoute
  val metricsPath: String = TestUtils.getMetricsOutputRoute

  before {
    sparkSession = TestUtils.getTestSession
  }

  after {
    sparkSession.stop()
  }


  it should "perform the full pipeline (mRMR Selection + Random Forest)" in {
    val numFeatures = 3
    val featureSelectionMethod = Constants.MRMR
    val classificationMethod = Constants.RANDOM_FOREST

    FeatureSelectionPipeline.run(sparkSession, testFile, outputFile, featureSelectionMethod, numFeatures)

    ClassificationPipeline.run(sparkSession, outputFile, metricsPath, modelsPath, classificationMethod)
  }
}
