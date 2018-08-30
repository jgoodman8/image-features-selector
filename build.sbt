name := "images-features-selection"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0"
libraryDependencies += "databricks" % "spark-deep-learning" % "1.0.0-spark2.3-s_2.11"
libraryDependencies += "org.bytedeco" % "javacv-platform" % "1.4.2"
libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.3.5"
