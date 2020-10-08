package ifs.jobs

import ifs.Constants.Selectors._
import ifs.services.{DataService, FeatureSelectionService, PreprocessService}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}
import org.apache.spark.sql.{DataFrame, SparkSession}

object FeatureSelectionPipeline extends App with Logging {

  def select(datasets: Array[DataFrame],
             features: String,
             labels: String,
             selectedFeatures: String,
             method: String, numFeatures: Int = 2): Array[DataFrame] = {

    method match {
      case CHI_SQ => FeatureSelectionService
        .selectWithChiSq(datasets, features, labels, selectedFeatures, numFeatures)
      case MRMR | MIM | MIFS | JMI | ICAP | CMIM | IF => FeatureSelectionService
        .selectWithInfoTheoretic(datasets, features, labels, selectedFeatures, method, numFeatures)
      case RELIEF => FeatureSelectionService
        .selectWithRelief(datasets, features, labels, selectedFeatures, numFeatures)
      case PCA => FeatureSelectionService.selectWithPCA(datasets, features, labels, selectedFeatures, numFeatures)
      case _ => throw new NoSuchMethodException("The feature selection method is not implemented")
    }
  }

  def preprocess(datasets: Array[DataFrame],
                 label: String,
                 features: String,
                 method: String): Array[DataFrame] = {
    method match {
      case PCA => PreprocessService.preprocessData(datasets, label, features)
      case CHI_SQ | RELIEF => PreprocessService.standardizeData(datasets, label, features)
      case MRMR | MIM | MIFS | JMI | ICAP | CMIM | IF => PreprocessService
        .preprocessAndDiscretize(datasets, label, features)
      case _ => throw new NoSuchMethodException("The feature selection method is not implemented")
    }
  }

  def run(session: SparkSession,
          trainFile: String,
          validationFile: String,
          testFile: String,
          outputPath: String,
          method: String,
          numFeatures: Int = 100,
          features: String = "features",
          label: String = "output_label",
          selectedFeatures: String = "selected"): Unit = {

    val hasValidationData = validationFile != null
    val datasets = PipelineUtils.load(session, trainFile, validationFile, testFile, hasValidationData)
    val preprocessedData = this.preprocess(datasets, label, features, method)
    val selectedDatasets = this.select(preprocessedData, features, label, selectedFeatures, method, numFeatures)

    if (hasValidationData) {
      DataService.save(selectedDatasets(0), fileDir = f"$outputPath%s/train")
      DataService.save(selectedDatasets(1), fileDir = f"$outputPath%s/val")
      DataService.save(selectedDatasets(2), fileDir = f"$outputPath%s/test")
    } else {
      DataService.save(selectedDatasets(0), fileDir = f"$outputPath%s/train")
      DataService.save(selectedDatasets(1), fileDir = f"$outputPath%s/test")
    }
  }

  val appName = args(0)
  val sparkSession = SparkSession.builder().appName(appName).getOrCreate()

  val sizeWithValidationFile = 7
  val sizeWithNoValidationFile = 6
  if (args.length == sizeWithValidationFile) {
    val Array(
    _,
    train: String,
    validation: String,
    test: String,
    output: String,
    method: String,
    numFeatures: String) = args
    this.run(sparkSession, train, validation, test, output, method, numFeatures.toInt)
  } else if (args.length == sizeWithNoValidationFile) {
    val Array(_, train: String, test: String, output: String, method: String, numFeatures: String) = args
    this.run(sparkSession, train, null, test, output, method, numFeatures.toInt)
  }

  sparkSession.stop()
}
