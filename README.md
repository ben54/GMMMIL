#Gaussian Mixture Model Image Playground 
This is an active learning playground GUI for doing image classification.  It uses scala and spark so those must be installed.  To run, just cd into the root directory and execute the following:

```
sbt run
```

I'm still working on the spark code.  Once that's done, the way the file will be executed will change - you'll have to build the code into a jar to submit to spark.  To build the "fat" jar, run 

```
sbt assembly
```

And, then to submit to spark, you'll run 

```
spark-submit --class clustering.LabelDemo --master local[*] target/scala-2.11/GMMMIL-assembly-1.0.jar
```

