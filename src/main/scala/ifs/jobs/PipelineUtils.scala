package ifs.jobs

import ifs.services.DataService
import org.apache.spark.sql.{DataFrame, SparkSession}

object PipelineUtils {
  def load(session: SparkSession,
           trainFile: String,
           validationFile: String,
           testFile: String,
           hasValidationData: Boolean): Array[DataFrame] = {

    val train: DataFrame = DataService.load(session, trainFile)
    val test: DataFrame = DataService.load(session, testFile)

    if (hasValidationData) {
      val validation: DataFrame = DataService.load(session, validationFile)

      return Array(train, validation, test)
    }

    Array(train, test)
  }
}
