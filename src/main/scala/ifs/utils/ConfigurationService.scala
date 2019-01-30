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

}
