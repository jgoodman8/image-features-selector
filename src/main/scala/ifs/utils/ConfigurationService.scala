package ifs.utils

import com.typesafe.config.{Config, ConfigFactory}

object ConfigurationService {
  val configuration: Config = ConfigFactory.load("application.conf")

  object Model {
    def getMaxIter: Int = configuration.getInt("Model.maxIter")

    def getElasticNetParam: Double = configuration.getDouble("Model.elasticNetParam")

    def getRegParam: Double = configuration.getDouble("Model.regParam")
  }

  object Data {
    def getTrainSplitRatio: Double = configuration.getDouble("DataSplit.train")

    def getTestSplitRatio: Double = configuration.getDouble("DataSplit.test")
  }

  object Session {
    def getDriverMaxResultSize: String = configuration.getString("Session.driverMaxResultSize")

    def getCheckpointDir: String = configuration.getString("Session.checkpointDir")

    def getModelDir: String = configuration.getString("Session.modelDir")

    def getOutputDir: String = configuration.getString("Session.outputDir")
  }

}
