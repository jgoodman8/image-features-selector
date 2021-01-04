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

      def getParallelism: Int = configuration.getInt("Model.linearSVC.parallelism")
    }

    object MLP {
      def getMaxIter: Int = configuration.getInt("Model.mlp.maxIter")

      def getBlockSize: Int = configuration.getInt("Model.mlp.blockSize")

      def getLayers: Array[Int] = {
        val layers = configuration.getIntList("Model.mlp.layers").toArray()

        layers.map(item => item.asInstanceOf[Int])
      }
    }

    object DecisionTree {
      def getMaxDepth: Int = {
        val configPath = "Model.dt.maxDepth"
        if (configuration.hasPath(configPath)) {
          configuration.getInt(configPath)
        } else {
          5
        }
      }

      def getMaxBins: Int = {
        val configPath = "Model.dt.maxBins"
        if (configuration.hasPath(configPath)) {
          configuration.getInt(configPath)
        } else {
          32 // Default value
        }
      }
    }

    object RandomForest {
      def getNumTrees: Int = {
        val configPath = "Model.rf.numTrees"
        if (configuration.hasPath(configPath)) {
          configuration.getInt(configPath)
        } else {
          20
        }
      }


      def getMaxDepth: Int = {
        val configPath = "Model.rf.maxDepth"
        if (configuration.hasPath(configPath)) {
          configuration.getInt(configPath)
        } else {
          5
        }
      }

      def getMaxBins: Int = {
        val configPath = "Model.rf.maxBins"
        if (configuration.hasPath(configPath)) {
          configuration.getInt(configPath)
        } else {
          32 // Default value
        }
      }

      def getSubsamplingRate: Double = {
        val configPath = "Model.rf.subsamplingRate"
        if (configuration.hasPath(configPath)) {
          configuration.getDouble(configPath)
        }

        1.0F // Default value
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

    def getNumTopAccuracy: Int = configuration.getInt("Session.numTopAccuracy")

    def getMaxCSVLength: Int = configuration.getInt("Session.maxCSVLength")
  }

}
