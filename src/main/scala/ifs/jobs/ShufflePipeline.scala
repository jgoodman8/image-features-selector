package ifs.jobs

import ifs.services.{ConfigurationService, DataService}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.rand

object ShufflePipeline extends App {

  def run(session: SparkSession, trainFile: String, testFile: String, outputPath: String): Unit = {

    var train: DataFrame = DataService.load(session, trainFile)
    var test: DataFrame = DataService.load(session, testFile)


    train = train.orderBy(rand())
    test = test.orderBy(rand())

    DataService.save(train, fileDir = f"$outputPath%s/train")
    DataService.save(test, fileDir = f"$outputPath%s/test")
  }

  val Array(appName: String, trainFile: String, testFile: String, output: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName(appName).getOrCreate()

  val metricsPath: String = ConfigurationService.Session.getMetricsPath
  val modelsPath: String = ConfigurationService.Session.getModelPath

  this.run(sparkSession, trainFile, testFile, output)

  sparkSession.close()
}
