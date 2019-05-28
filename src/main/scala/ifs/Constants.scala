package ifs

object Constants {

  object Classifiers {
    val LOGISTIC_REGRESSION: String = "lr" // Logistic Regression
    val RANDOM_FOREST: String = "rf" // Random Forest
    val DECISION_TREE: String = "tree" // Decision Tree
    val MLP: String = "mlp" // Multi Layer Perceptron
    val NAIVE_BAYES: String = "nb" // Naive Bayes
    val SVM: String = "svm" // Support Vector Machine with Linear Kernel
  }

  object Selectors {
    val CHI_SQ: String = "chisq" // Chi-squared
    val MRMR: String = "mrmr" // Minimum Redundancy Maximum Relevance
    val MIM: String = "mim" // Mutual Information Maximization
    val MIFS: String = "mifs" // Mutual Information Feature Selection
    val JMI: String = "jmi" // Joint Mutual Information
    val ICAP: String = "icap" // Interaction Capping
    val CMIM: String = "cmim" // Conditional Mutual Information Maximization
    val IF: String = "if" // Informative Fragments
    val RELIEF: String = "relief" // RELIEF-F
    val PCA: String = "pca" // Principal Component Analysis
  }

}
