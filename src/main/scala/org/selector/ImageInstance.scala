package org.selector

import org.apache.spark.ml.image.ImageSchema._
import org.apache.spark.sql.Row

class ImageInstance(tableRow: Row) {

  val imageRow: Row = tableRow.getAs[Row](0)
  val label: String = tableRow.getString(1)

  val origin: String = getOrigin(imageRow)
  val height: Int = getHeight(imageRow)
  val width: Int = getWidth(imageRow)
  val nChannels: Int = getNChannels(imageRow)
  val mode: Int = getMode(imageRow)
  val data: Array[Byte] = getData(imageRow)

  def asTuple: (Row, String) = {
    (imageRow, label)
  }
}
