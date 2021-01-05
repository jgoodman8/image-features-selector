package ifs

import ifs.jobs.ClassificationPipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class Execution extends FlatSpec {

  def getMetricsOutputRoute: String = "output/metrics"

  def getModelsRoute: String = "output/models"

  "execution" should "run" in {
    val session = SparkSession.builder()
      .appName("Execution")
      .master("local[4]")
      .config("spark.driver.memory", "12g")
      .config("spark.executor.memory", "12g")
      .config("spark.driver.maxResultSize", "5g")
      .config("spark.driver.cores", "14")
      .config("spark.driver.memory", "12g")
      .config("spark.shuffle.spill", "false")
      .getOrCreate()

    ClassificationPipeline.run(
      session,
      "/home/jgoodman/data/oxford/features/lbp/g4p8r1/train.csv",
      "/home/jgoodman/data/oxford/features/lbp/g4p8r1/val.csv",
      "/home/jgoodman/data/oxford/features/lbp/g4p8r1/test.csv",
      getMetricsOutputRoute,
      getModelsRoute,
      Constants.Classifiers.RANDOM_FOREST
    )

  }

  "vgg" should "classify fucking awesome" in {
    val session = SparkSession.builder()
      .appName("Execution")
      .master("local[4]")
      .config("spark.driver.memory", "12g")
      .config("spark.executor.memory", "12g")
      .config("spark.driver.maxResultSize", "5g")
      .config("spark.driver.cores", "14")
      .config("spark.driver.memory", "12g")
      .config("spark.shuffle.spill", "false")
      .getOrCreate()

    ClassificationPipeline.run(
      session,
      "/home/jgoodman/data/oxford/features/vgg19/train.csv",
      "/home/jgoodman/data/oxford/features/vgg19/val.csv",
      "/home/jgoodman/data/oxford/features/vgg19/test.csv",
      getMetricsOutputRoute,
      getModelsRoute,
      Constants.Classifiers.RANDOM_FOREST
    )

  }
}