package clustering

import scala.collection.immutable.Vector
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Multinomial
import breeze.linalg.DenseVector

class GMM(alpha: Vector[Double], means: Vector[Double], std: Vector[Double]) {
  assert(alpha.length == means.length && alpha.length == std.length, "alpha vector must have same length as mean vector and std vector")
  val alphaMultinomial = new Multinomial(DenseVector(alpha.toArray))
  val numComponents = alpha.length
  val components: Vector[Gaussian] = means.zip(std).map(meanStd => new Gaussian(meanStd._1,meanStd._2))
  def f(x: Double): Double = {
//    val t = alpha.zip(components)
    alpha.zip(components).map(alphaComponent => alphaComponent._1 * alphaComponent._2.pdf(x)).sum
  }
  def draw: Double = {
    //sample from alpha to get a component
    val compIndex = alphaMultinomial.draw
    components(compIndex).draw()
  }
  
  def likelihood(data: Vector[Double]): Vector[Double] = data.map(datum => f(datum))
  
  def logProbDataAndClass(data: Vector[Double], classProb: Double): Double = {
//    data.map(datum => Math.log(f(datum)*classProb)).sum
    data.map(datum => Math.log(f(datum))).sum + Math.log(classProb)
  }
 
}

object GMM {
  private val rand = new scala.util.Random
//  def fit(data: Vector[Double], numComponents: Int): GMM = {
//    
//  }
}

class MixtureOfGMMs(prior: Vector[Double], alphas: Vector[Vector[Double]], means: Vector[Vector[Double]], stds: Vector[Vector[Double]]) {
  assert(prior.length == alphas.length && prior.length == means.length && prior.length == stds.length, "all parameter vectors must be of the same length")
  val priorMultinomial = new Multinomial(DenseVector(prior.toArray)) 
  val numGMMs = prior.length
  val GMMs: Vector[GMM] = (alphas, means, stds).zipped.toVector.map(params => new GMM(params._1, params._2, params._3))
  
  //returns vector of (class, bag of data)
  def draw(numBags: Int, bagSize: Int): Vector[(Int,Vector[Double])] = {
    //sample from prior
    val classSamples = 0.until(numBags).toVector.map(index => priorMultinomial.draw)
    //sample from gmm corresponding to above sample
    val samples: Vector[(Int,Vector[Double])] = classSamples.map(classIndex => (classIndex,Vector.fill(bagSize)(GMMs(classIndex).draw)))
    samples
  }
  
  def classify(bag: Vector[Double]): Int = {
    //for each GMM, calculate log likelihood of data under that distribution * prior and max over this
    val t = GMMs.zip(prior).map(gmmPrior => gmmPrior._1.logProbDataAndClass(bag, gmmPrior._2)).zipWithIndex.maxBy(x => x._1)._2
    GMMs.zip(prior).map(gmmPrior => gmmPrior._1.logProbDataAndClass(bag, gmmPrior._2)).zipWithIndex.maxBy(_._1)._2
  }
}