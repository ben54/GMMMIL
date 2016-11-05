#Gaussian Mixture Model Image Playground 
This is an active learning playground GUI for doing image classification.  It uses scala and spark so those must be installed.  To run, just cd into the root directory and execute the following:

```
sbt assembly
spark-submit --class clustering.LabelDemo --master local[*] target/scala-2.11/GMMMIL-assembly-1.0.jar
```

