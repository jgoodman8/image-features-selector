package ifs.strategies

import ifs.services.FeatureSelectionService
import ifs.Constants.Selectors._
import org.apache.spark.sql.DataFrame

object SelectionStrategy {
  def select_by_method(train: DataFrame, test: DataFrame, features: String, labels: String, selectedFeatures: String,
                       method: String, numFeatures: Int = 2): Array[DataFrame] = {

    method match {
      case CHI_SQ => FeatureSelectionService
        .selectWithChiSq(train, test, features, labels, selectedFeatures, numFeatures)
      case MRMR | MIM | MIFS | JMI | ICAP | CMIM | IF => FeatureSelectionService
        .selectWithInfoTheoretic(train, test, features, labels, selectedFeatures, method, numFeatures)
      case RELIEF => FeatureSelectionService
        .selectWithRelief(train, test, features, labels, selectedFeatures, numFeatures)
      case PCA => FeatureSelectionService.selectWithPCA(train, test, features, labels, selectedFeatures, numFeatures)
      case _ => throw new NoSuchMethodException("The feature selection method is not implemented")
    }
  }
}
