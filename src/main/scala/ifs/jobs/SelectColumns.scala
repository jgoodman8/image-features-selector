package ifs.jobs

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

import java.io.File

object SelectColumns extends App {
  def run(session: SparkSession, src: String, dst: String, targetFeaturesTop33: Int, targetFeaturesTop10: Int): Unit = {

    val dstPath = new File(dst)
    val dstFolder = dstPath.getParentFile.getName
    val dstFilename = dstPath.getName

    //   Load data (from top50)
    val data: DataFrame = session.read.csv(src)

    //   Store top50 in one CSV
    data.coalesce(1).write.mode(SaveMode.Overwrite).csv(dstFolder + "/top50/" + dstFilename)

    //   Select top33 and in one CSV
    val selectedTop33Columns = (data.columns.take(targetFeaturesTop33) :+ data.columns.tail)
      .map(colName => col(colName.toString))

    data.select(selectedTop33Columns: _*)
      .coalesce(1)
      .write.mode(SaveMode.Overwrite).csv(dstFolder + "/top33/" + dstFilename)

    //    Select top10 and in one CSV
    val selectedTop10Columns = (data.columns.take(targetFeaturesTop10) :+ data.columns.tail)
      .map(colName => col(colName.toString))

    data.select(selectedTop10Columns: _*)
      .coalesce(1)
      .write.mode(SaveMode.Overwrite).csv(dstFolder + "/top10/" + dstFilename)

  }

  val Array(src: String, dst: String, targetFeaturesTop33: String, targetFeaturesTop10: String) = args

  val sparkSession: SparkSession = SparkSession.builder().appName("SelectColumns").getOrCreate()

  this.run(sparkSession, src, dst, Integer.parseInt(targetFeaturesTop33), Integer.parseInt(targetFeaturesTop10))

  sparkSession.close()

}
