package ifs.strategies

import ifs.Constants.Classifiers._
import ifs.Constants.Selectors._
import ifs.services.PreprocessService
import org.apache.spark.sql.DataFrame

object PreprocessStrategy {
  def preprocess_for_training(train: DataFrame, test: DataFrame, label: String, features: String,
                              method: String): Array[DataFrame] = {
    method match {
      case LOGISTIC_REGRESSION | RANDOM_FOREST | DECISION_TREE | MLP => PreprocessService
        .preprocessData(train, test, label, features)
      case NAIVE_BAYES | SVM => PreprocessService.preprocessAndScaleData(train, test, label, features)
      case _ => throw new NoSuchMethodException("The classifier method is not implemented")
    }
  }

  def preprocess_for_feature_selection(train: DataFrame, test: DataFrame, label: String, features: String,
                                       method: String): Array[DataFrame] = {
    method match {
      case PCA => PreprocessService.preprocessData(train, test, label, features)
      case CHI_SQ | RELIEF => PreprocessService.standardizeData(train, test, label, features)
      case MRMR | MIM | MIFS | JMI | ICAP | CMIM | IF => PreprocessService
        .preprocessAndDiscretize(train, test, label, features)
      case _ => throw new NoSuchMethodException("The feature selection method is not implemented")
    }
  }
}
