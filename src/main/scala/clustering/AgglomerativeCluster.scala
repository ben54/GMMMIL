package clustering

import breeze.linalg.DenseVector

class AgglomerativeCluster(root: Group) {
  lazy val size = root.size
  def cutAt(numClusters: Int): Vector[Group] = {
    assert(numClusters < size, "You asked for " + numClusters + " clusters, but there are only " + size + " data points")
    var groups = Vector(root)
    while (groups.length < numClusters) {
      val minGroup = groups.maxBy(group => group.distance)
      groups = groups.filter(group => group != minGroup) ++ minGroup.expand
    }
    groups
  }
}

object AgglomerativeCluster {
  def apply(data: Vector[DenseVector[Double]], metric: (Group, Group) => Double): AgglomerativeCluster = {
    //first, put all data into a singleton
//    val singletons: Vector[Singleton] = data.map(datum => new Singleton(datum))
    var groups: Vector[Group] = data.map(datum => new Singleton(datum))
    while (groups.length > 1) {
      //find min pair-wise distance
      val distances: Vector[Vector[(Group,Group,Double)]] = groups.map{ group_i => groups.map{ group_j => 
          (group_i, group_j, if (group_i == group_j) Double.MaxValue else metric(group_i, group_j))
        }
      }
      val minDistances: Vector[(Group,Group,Double)] = distances.map(pair => pair.minBy(_._3))
      val minPair: (Group,Group,Double) = minDistances.minBy(_._3)
      groups = groups.filter(group => group != minPair._1 && group != minPair._2) :+ new Doubleton((minPair._1, minPair._2), minPair._3)
    }
  new AgglomerativeCluster(groups(0))
  }
  
  def singleLink(distance: (DenseVector[Double], DenseVector[Double]) => Double)(g1: Group, g2: Group): Double = {
    g1.getElements.flatMap(element1 => g2.getElements.map(element2 => distance(element1, element2))).min 
  }
  
  def completeLink(distance: (DenseVector[Double], DenseVector[Double]) => Double)(g1: Group, g2: Group): Double = {
    g1.getElements.flatMap(element1 => g2.getElements.map(element2 => distance(element1, element2))).max 
  }
  
  def averageLink(distance: (DenseVector[Double], DenseVector[Double]) => Double)(g1: Group, g2: Group): Double = {
    val numElements: Double = g1.getElements.length * g2.getElements.length
    g1.getElements.flatMap(element1 => g2.getElements.map(element2 => distance(element1, element2))).sum / numElements 
  }
  
  def centroidLink(distance: (DenseVector[Double], DenseVector[Double]) => Double)(g1: Group, g2: Group): Double = {
    val length1 = g1.getElements.length
    val length2 = g2.getElements.length
    val centroid1 = DenseVector(g1.getElements.map(e => e.toScalaVector()).transpose.map(dim => dim.sum / length1).toArray)
    val centroid2 = DenseVector(g2.getElements.map(e => e.toScalaVector()).transpose.map(dim => dim.sum / length2).toArray)
    distance(centroid1, centroid2) 
  }
  
  def euclideanDistance(vec1: DenseVector[Double], vec2: DenseVector[Double]): Double = {
    assert(vec1.length == vec2.length, "Vectors must be of same length")
    Math.sqrt(vec1.toScalaVector.zip(vec2.toScalaVector).map(pair => Math.pow(pair._1 - pair._2,2)).sum)
  }
  
  def manhattanDistance(vec1: DenseVector[Double], vec2: DenseVector[Double]): Double = {
    assert(vec1.length == vec2.length, "Vectors must be of same length")
    vec1.toScalaVector.zip(vec2.toScalaVector).map(pair => Math.abs(pair._1 - pair._2)).sum 
  }
}

trait Group {
  def size: Double
  def getElements: Vector[DenseVector[Double]]
  def distance: Double
  def expand: Vector[Group]
}

class Doubleton(subgroups: (Group, Group), interGroupDistance: Double) extends Group{ 
  lazy val size = subgroups._1.size + subgroups._2.size
  lazy val getElements: Vector[DenseVector[Double]] = subgroups._1.getElements ++ subgroups._2.getElements
  def distance = interGroupDistance
  lazy val expand = Vector(subgroups._1, subgroups._2)
} 

//object Doubleton {
//  def apply(subgroups: (Group, Group)): Doubleton = new Doubleton(subgroups)
//}

class Singleton(element: DenseVector[Double]) extends Group{
  def size = 1
  lazy val getElements: Vector[DenseVector[Double]] = Vector(element)
  def distance = 0
  lazy val expand = Vector(this)
}

//object Singleton {
//  def apply(element: Vector[Double]) = new Singleton(element)
//}





