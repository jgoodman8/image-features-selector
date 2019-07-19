package ifs.jobs

import ifs.services.DataService
import ifs.strategies.{PreprocessStrategy, SelectionStrategy}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession}

object FeatureSelectionPipeline extends App with Logging {

  def run(session: SparkSession, trainFile: String, testFile: String, outputPath: String, method: String,
          numFeatures: Int = 100, features: String = "features", label: String = "output_label",
          selectedFeatures: String = "selected"): Unit = {

    val train: DataFrame = DataService.load(session, trainFile)
    val test: DataFrame = DataService.load(session, testFile)

    val Array(preprocessedTrain, preprocessedTest) = PreprocessStrategy
      .preprocess_for_feature_selection(train, test, label, features, method)

    val Array(selectedTrain, selectedTest) = SelectionStrategy
      .select_by_method(preprocessedTrain, preprocessedTest, features, label, selectedFeatures, method, numFeatures)

    DataService.save(selectedTrain, fileDir = f"$outputPath%s/train", label, selectedFeatures)
    DataService.save(selectedTest, fileDir = f"$outputPath%s/test", label, selectedFeatures)
  }

  val Array(appName: String, train: String, test: String, output: String, method: String, numFeatures: String) = args

  val sparkSession = SparkSession.builder().appName(appName).getOrCreate()

  this.run(sparkSession, train, test, output, method, numFeatures.toInt)

  sparkSession.stop()
}
