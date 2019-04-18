package ifs.services

import com.typesafe.config.{Config, ConfigFactory}

object ConfigurationService {
  val configuration: Config = ConfigFactory.load("application.conf")

  object Model {
    def getMaxIter: Int = configuration.getInt("Model.maxIter")

    def getElasticNetParam: Double = configuration.getDouble("Model.elasticNetParam")

    def getRegParam: Double = configuration.getDouble("Model.regParam")

    def getMetrics: Array[String] = Array("accuracy")
  }

  object Preprocess {
    def getNumberOfBeans: Int = configuration.getInt("Preprocess.beans")

    def getMinScaler: Int = configuration.getInt("Preprocess.min")

    def getMaxScaler: Int = configuration.getInt("Preprocess.max")
  }

  object Session {
    def getDriverMaxResultSize: String = configuration.getString("Session.driverMaxResultSize")

    def getCheckpointDir: String = configuration.getString("Session.checkpointDir")

    def getModelPath: String = configuration.getString("Session.modelPath")

    def getMetricsPath: String = configuration.getString("Session.metricsPath")
  }

}
