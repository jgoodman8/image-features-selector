name := "images-features-selection"

version := "0.1"

scalaVersion := "2.11.8"
val sparkVersion = "2.2.2"

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
libraryDependencies += "databricks" % "spark-deep-learning" % "1.5.0-spark2.4-s_2.11" % "provided"
libraryDependencies += "org.bytedeco" % "javacv-platform" % "1.4.2"
libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.3.5"
libraryDependencies += "com.typesafe" % "config" % "1.3.3"
libraryDependencies += "sramirez" % "spark-infotheoretic-feature-selection" % "1.4.4" exclude("io.netty", "netty") exclude("commons-net", "commons-net") exclude("com.google.guava", "guava")  exclude("org.apache.spark", "spark-mllib")

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.5"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"


test in assembly := {}

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}