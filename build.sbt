name := "GMMMIL"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVers = "2.0.0"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value

libraryDependencies += "org.scalaz" %% "scalaz-core" % "7.2.5"

libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.0"

libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-io-extra" % "2.1.0"

libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-filters" % "2.1.0"

// https://mvnrepository.com/artifact/org.scalanlp/breeze_2.11
libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.13-RC1"

// https://mvnrepository.com/artifact/org.scalanlp/breeze-natives_2.11
libraryDependencies += "org.scalanlp" % "breeze-natives_2.11" % "0.13-RC1"

// https://mvnrepository.com/artifact/org.scalanlp/breeze-viz_2.11
libraryDependencies += "org.scalanlp" % "breeze-viz_2.11" % "0.13-RC1"


// https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.11
libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.0.1" % "provided"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib_2.11
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.0.1" % "provided"

// https://mvnrepository.com/artifact/org.scala-lang/scala-swing
libraryDependencies += "org.scala-lang" % "scala-swing" % "2.11.0-M7"

// https://mvnrepository.com/artifact/com.azavea.geotrellis/geotrellis-raster_2.11
libraryDependencies += "com.azavea.geotrellis" % "geotrellis-raster_2.11" % "0.10.3"

// https://mvnrepository.com/artifact/com.azavea.geotrellis/geotrellis-engine_2.11
libraryDependencies += "com.azavea.geotrellis" % "geotrellis-engine_2.11" % "0.10.3"

resolvers += "stephenjudkins-bintray" at "http://dl.bintray.com/stephenjudkins/maven"

libraryDependencies += "ps.tricerato" %% "pureimage" % "0.1.2"