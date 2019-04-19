package ifs.jobs

import ifs.Constants._
import ifs.services.{DataService, FeatureSelectionService, PreprocessService}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession}

object FeatureSelectionPipeline extends App with Logging {

  def select(train: DataFrame, test: DataFrame, features: String, labels: String, selectedFeatures: String,
             method: String, numFeatures: Int = 2): Array[DataFrame] = {

    method match {
      case CHI_SQ => FeatureSelectionService
        .selectWithChiSq(train, test, features, labels, selectedFeatures, numFeatures)
      case MRMR | MIM | MIFS | JMI | ICAP | CMIM | IF => FeatureSelectionService
        .selectWithInfoTheoretic(train, test, features, labels, selectedFeatures, method, numFeatures)
      case RELIEF => FeatureSelectionService
        .selectWithRelief(train, test, features, labels, selectedFeatures, numFeatures)
      case _ => throw new NoSuchMethodException("The feature selection method is not implemented")
    }
  }

  def preprocess(train: DataFrame, test: DataFrame, label: String, features: String,
                 method: String): Array[DataFrame] = {
    method match {
      case CHI_SQ | RELIEF => PreprocessService.preprocessData(train, test, label, features)
      case MRMR | MIM | MIFS | JMI | ICAP | CMIM | IF => PreprocessService
        .preprocessAndDiscretize(train, test, label, features)
      case _ => throw new NoSuchMethodException("The feature selection method is not implemented")
    }
  }

  def run(session: SparkSession, trainFile: String, testFile: String, outputPath: String, method: String,
          numFeatures: Int = 100, features: String = "features", label: String = "output_label",
          selectedFeatures: String = "selected"): Unit = {

    val train: DataFrame = DataService.load(session, trainFile)
    val test: DataFrame = DataService.load(session, testFile)

    val Array(preprocessedTrain, preprocessedTest) = this.preprocess(train, test, label, features, method)

    val Array(selectedTrain, selectedTest) = this
      .select(preprocessedTrain, preprocessedTest, features, label, selectedFeatures, method, numFeatures)

    DataService.save(selectedTrain, fileDir = f"$outputPath%s/train", label, selectedFeatures)
    DataService.save(selectedTest, fileDir = f"$outputPath%s/test", label, selectedFeatures)
  }

  val Array(appName: String, train: String, test: String, output: String, method: String, numFeatures: String) = args

  val sparkSession = SparkSession.builder().appName(appName).getOrCreate()

  this.run(sparkSession, train, test, output, method, numFeatures.toInt)

  sparkSession.stop()
}
