package ifs.services

import com.typesafe.config.{Config, ConfigFactory}

object ConfigurationService {
  val configuration: Config = ConfigFactory.load("application.conf")

  object Model {

    def getNumFolds: Int = configuration.getInt("Model.numFolds")

    def isGridSearchActivated: Boolean = configuration.getBoolean("Model.gridSearch")

    object LogisticRegression {
      def getMaxIter: Int = configuration.getInt("Model.logisticRegression.maxIter")

      def getElasticNetParam: Double = configuration.getDouble("Model.logisticRegression.elasticNetParam")

      def getRegParam: Double = configuration.getDouble("Model.logisticRegression.regParam")
    }

    object LinearSVC {
      def getMaxIter: Int = configuration.getInt("Model.linearSVC.maxIter")

      def getRegParam: Double = configuration.getDouble("Model.linearSVC.regParam")
    }

    object MLP {
      def getMaxIter: Int = configuration.getInt("Model.mlp.maxIter")

      def getBlockSize: Int = configuration.getInt("Model.mlp.blockSize")
    }

    object RandomForest {
      def getNumTrees: Int = configuration.getInt("Model.rf.numTrees")

      def hasMaxDepth: Boolean = configuration.hasPath("Model.rf.maxDepth")

      def getMaxDepth: Int = configuration.getInt("Model.rf.maxDepth")

      def getSubsamplingRate: Double = {
        val configPath = "Model.rf.subsamplingRate"
        if (configuration.hasPath(configPath)) {
          configuration.getDouble(configPath)
        }

        1.0F
      }
    }

    def getMetrics: Array[String] = Array("accuracy", "f1")
  }

  object Preprocess {

    object Discretize {
      def getNumberOfBeans: Int = configuration.getInt("Preprocess.discretize.beans")
    }

    object Scale {
      def getMinScaler: Int = configuration.getInt("Preprocess.scale.min")

      def getMaxScaler: Int = configuration.getInt("Preprocess.scale.max")
    }

  }

  object FeatureSelection {

    object InfoTheoretic {
      def getNumberOfPartitions: Int = configuration.getInt("FeatureSelection.infoTheoretic.partitions")
    }

    object Relief {
      def getEstimationRatio: Double = configuration.getDouble("FeatureSelection.relief.estimationRatio")

      def getNumberOfNeighbors: Int = configuration.getInt("FeatureSelection.relief.numNeighbors")

      def isDiscreteData: Boolean = configuration.getBoolean("FeatureSelection.relief.discreteData")
    }

  }

  object Session {
    def getDriverMaxResultSize: String = configuration.getString("Session.driverMaxResultSize")

    def getCheckpointDir: String = configuration.getString("Session.checkpointDir")

    def getModelPath: String = configuration.getString("Session.modelPath")

    def getMetricsPath: String = configuration.getString("Session.metricsPath")
  }

}
