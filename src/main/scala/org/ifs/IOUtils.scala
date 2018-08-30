package org.ifs

import com.github.tototoshi.csv._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row}

object IOUtils {

  /**
    * Saves a given DataFrame with labels and features as a csv file
    *
    * @param data     DataFrame with labels and features
    * @param fileName Name to the resulting file
    */
  def exportToCSV(data: DataFrame, fileName: String): Unit = {
    val rows: Array[Row] = data.collect()

    val writer = CSVWriter.open(fileName, append = true)

    rows.foreach(row => {
      val features: Vector[Double] = row.get(0).asInstanceOf[DenseVector].toArray.toVector
      val label: Int = row.getInt(1)
      val newRow = features :+ label

      writer.writeRow(newRow.toList)
    })

    writer.close()
  }
}
