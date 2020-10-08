package org.apache.spark.ml.param

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

@DeveloperApi
class DatasetParam(parent: String, name: String, doc: String, isValid: Dataset[_] => Boolean)
  extends Param[Dataset[_]](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, ParamValidators.alwaysTrue)

  def this(parent: Identifiable, name: String, doc: String, isValid: Dataset[_] => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: Dataset[_]): ParamPair[Dataset[_]] = super.w(value)

  //  override def jsonEncode(value: Dataset[_]): String = {
  //    compact(render(value.toJSON.))
  //  }
  //
  //  override def jsonDecode(json: String): Double = {
  //    DoubleParam.jValueDecode(parse(json))
  //  }
}
