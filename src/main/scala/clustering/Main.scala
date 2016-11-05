package clustering

import java.io.BufferedInputStream
import java.io.FileInputStream
import com.sksamuel.scrimage.{Image => sImage}
import com.sksamuel.scrimage.ScaleMethod
import com.sksamuel.scrimage._
import com.sksamuel.scrimage.composite._
import com.sksamuel.scrimage.{Color => sColor}
import java.io.File
import scala.collection.immutable.Vector
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Multinomial
import breeze.linalg.DenseVector
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.JScrollPane
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.awt.Graphics
import java.awt.image.BufferedImage
import javax.swing.AbstractButton
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.ImageIcon
import java.awt.event.ActionEvent
import java.awt.event.ActionListener
import java.awt.event.KeyEvent
import breeze.plot._
import scala.swing._
import geotrellis.raster.io.geotiff.reader
import geotrellis.raster.io.geotiff.MultibandGeoTiff
import geotrellis.raster._
import geotrellis.raster.io.geotiff._
import geotrellis.raster.render._
import com.typesafe.config.ConfigFactory
import scala.swing.event.ButtonClicked
import javax.swing.{ Box, BoxLayout, JDialog, JFrame, ImageIcon, UIManager }
import java.awt.{ BorderLayout, Component, Dimension, Graphics, Point, Toolkit }
//import java.awt.Color
import java.awt.image.BufferedImage
import java.net.URL
//import ps.tricerato.pureimage._

object Main {
  
  val windowWidth = 100
	val windowHeight = 100
	val numWindows = 20
	
	val image: sImage = Image.fromFile(new File("data/image.png"))
	val sampledWindow = image.subimage(0, 0, windowWidth, windowHeight)
		
//	val c: sColor = sColor(1,3,5,10)//r,g,b,alpha as Int
//	def threshold(x: Int, y: Int, p: Pixel):Pixel = {
//    p.argb.red match {
//      case x if 0 until 10 contains x => 100
//      case _ => 10 
//    }
//    
//  }
//  
//	image.map((Int,Int,com.sksamuel.scrimage.Pixel) => )
	val rand = scala.util.Random
  rand.setSeed(200)
	
//	def main(args: Array[String]) {

//		val conf = new SparkConf().setAppName("GMMMIL")
//    val sc = new SparkContext(conf)
		
//	  displayWindow
    
//    convertGeotiffToPng("data/image.tif","data/image.png")
    
    
//	  image.output("test.png")
//		println("w: " + image.width + ", h: " + image.height)
		
		//create RDD of window pixels
//		val patchPixels = sc.parallelize(image.subimage(0, 0, 100, 100).pixels.toVector.map(pixel => Vectors.dense(pixel.red.toDouble, pixel.green.toDouble, pixel.blue.toDouble)))
//    val gmm = new GaussianMixture().setConvergenceTol(.1).run(patchPixels)
//    val gmm = new GaussianMixture().setK(2).run(parsedData)
//	}
  
//  def getLabeledGMMs: Vector[(GMM,Int)] = {
//    
//  }
  
  def writeImagesToFile(xyImage: Vector[(Int,Int,sImage)]) = {
//    image.subimage(0, 0, windowWidth, windowHeight).output("data/output.tif")
    xyImage.foreach(window => window._3.output("data/output" + window._1 + "_" + window._2 + ".tif"))
  }
  
  def convertGeotiffToPng() = {
    val image: sImage = Image.fromFile(new File("data/image.png"))
		val geo2 = reader.GeoTiffReader.readMultiband("data/image.tif").convert(DoubleConstantNoDataCellType)
	  geo2.renderPng().write("data/image.png")
  }
  
  def sampleImagesToLabel() = {
    //produce images to label
		val labels = Vector(0,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1)
		val x = 0.until(numWindows).map(x => rand.nextInt(image.width - windowWidth))
		val y = 0.until(numWindows).map(y => rand.nextInt(image.height - windowHeight))
		val coords: Vector[(Int,Int)] = x.zip(y).toVector
		val windows: Vector[(Int,Int,sImage)] = coords.map(coord => (coord._1, coord._2, image.subimage(coord._1,coord._2,windowWidth, windowHeight)))
  }
  
  
  
}

object Classifier {
  val rural = 0
  val urban = 1
  val mixtureOfGMM = 1
  val graphClustering = 2
}

//class Classifier(image: sImage, sc: SparkContext, wWidth: Int, wHeight: Int) {
class Classifier(image: sImage, wWidth: Int, wHeight: Int) {
  private var imgCopy = image.copy
  def windowWidth = wWidth
  def windowHeight = wHeight
//  val image = img
  //map from (x,y) -> label using Classifier.rural / Classifier.urban
  var labeled: scala.collection.mutable.Map[(Int,Int),Int] = scala.collection.mutable.Map()
  def getClassPrior: Map[Int,Double] = {
    
    val prior = labeled.map(xyLabel => xyLabel._2).toVector.groupBy(x => x).map(labelLabelVec => (labelLabelVec._1, labelLabelVec._2.length / labeled.size.toDouble)).toMap
    println(prior.map(labelProb => labelProb._1 + ": " + labelProb._2).mkString(","))
    prior
  }
//  def getLocalGMMs: Vector[(GaussianMixtureModel,Int)] = {
//    //build local gmms
//    val patchPixels: Vector[(RDD[org.apache.spark.mllib.linalg.Vector],Int)] = labeled.toVector.map{xyLabel => 
//      (sc.parallelize(image.subimage(xyLabel._1._1, xyLabel._1._2, windowWidth, windowHeight).pixels.toVector.map {pixel => 
//        Vectors.dense(pixel.red.toDouble, pixel.green.toDouble, pixel.blue.toDouble)
//      }),xyLabel._2)
//    }
//    val gmmsLabels = patchPixels.map(patchLabel => (new GaussianMixture().setConvergenceTol(.1).run(patchLabel._1), patchLabel._2))
//    gmmsLabels
//  }
  def labeledImage: sImage = {
    labeled.foreach{ xyLabel =>
      val pix = if (xyLabel._2 == Classifier.urban) Pixel(255, 0, 0, 255) else Pixel(0,255,0,255)
      for (i <- 0 until windowWidth) {
        for (j <- 0 until windowHeight) {
          imgCopy.setPixel(xyLabel._1._1 + i, xyLabel._1._2 + j, pix)
        }
      }
    }
    imgCopy
  }
//  def logLikelihoodOfPatch(data: Array[Pixel], gmm: GaussianMixtureModel, labelPrior: Double): Double = {
//    val vectors = sc.parallelize(data.toVector.map(pixel => Vectors.dense(pixel.red.toDouble, pixel.green.toDouble, pixel.blue.toDouble)))
//    gmm.predictSoft(vectors).map(fi => Math.log(fi.zip(gmm.weights).map(pair => pair._1 * pair._2).sum)).sum() + Math.log(labelPrior)
//  }
//  def predictedImage: sImage = {
//    val img = image.copy
//    val priors: Map[Int,Double] = getClassPrior
//    val gmmLabels: Vector[(GaussianMixtureModel,Int)] = getLocalGMMs
//    val stepSize = 300
//    for (i <- 0 until image.width by stepSize) {
//      for (j <- 0 until image.height by stepSize) {
//        val patch = img.pixels(i, j, Math.min(image.width -1 - i, stepSize), Math.min(image.height -1 - j, stepSize))
//        val probLabels: Vector[(Double,Int)] = gmmLabels.map(gmmLabels => (logLikelihoodOfPatch(patch, gmmLabels._1, priors(gmmLabels._2)), gmmLabels._2))
//        val mostLikelyLabel = probLabels.maxBy(_._1)._2
//        for (x <- i until Math.min(image.width -1, i + stepSize)) {
//          for (y <- j until Math.min(image.height -1, j + stepSize)) {
//            val p = img.pixel(x,y)
//            if (mostLikelyLabel == Classifier.urban) img.setPixel(x,y, Pixel(p.red*2, p.green, p.blue, p.alpha)) else img.setPixel(x,y, Pixel(p.red, p.green*2, p.blue, p.alpha)) 
//          }
//        }
//      }
//    }
//    img
//  }
  
//  def getImagePatchs(stepSize: Int): RDD[Vector[(Int,Int,Pixel)]] = {
//    val xs = 0 until image.width by stepSize
//    val ys = 0 until image.height by stepSize
//    val widths = Vector.fill(xs.length - 1)(stepSize) ++ Vector((image.width -1) % stepSize)
//    val heights = Vector.fill(ys.length - 1)(stepSize) ++ Vector((image.height -1) % stepSize)
//    var windows: List[(Int,Int,Int,Int)] = List()
//    for (i <- 0 until xs.length) {
//      for (j <- 0 until ys.length) {
//        (xs(i),ys(i),widths(i),heights(i)) :: windows
//      }
//    }
//    val patches = windows.map(tuple => (tuple._1, tuple._2, image.patch(tuple._1, tuple._2, tuple._3, tuple._4)))
//    
//  }
}


class LearnerGUI extends GridPanel(1, 2) {
//  val conf = new SparkConf().setAppName("GMMMIL")
//  val sc = new SparkContext(conf)
  
  val windowWidth = 100
  val windowHeight = 100
  var currentX = 0 //get swapped out during learning - used to choose patch
  var currentY = 0
  
  
//  val labeled: scala.collection.mutable.Map[(Int,Int),Int] = scala.collection.mutable.Map()
  
  val r = new scala.util.Random
  r.setSeed(200)
  
  val pureImage: sImage = Image.fromFile(new File("data/image.png"))
  val image = pureImage.copy
//  val classifier = new Classifier(image, sc, windowWidth, windowHeight)
  val classifier = new Classifier(image, windowWidth, windowHeight)
  val patch = image.subimage(currentX, currentY, windowWidth, windowHeight)
//  val patchRed = image.subimage(0, 0, windowWidth, windowHeight).map{(x: Int, y: Int, p: Pixel) => 
//	  Pixel(p.red, p.green * 2, p.blue, p.alpha)
//	}
  
  val imageIcon: ImageIcon = Swing.Icon(image.scale(0.1, ScaleMethod.Bicubic).toNewBufferedImage(BufferedImage.TYPE_INT_ARGB))
  val patchIcon: ImageIcon = Swing.Icon(patch.scale(3.0, ScaleMethod.Bicubic).toNewBufferedImage(BufferedImage.TYPE_4BYTE_ABGR))
  
  //Create the first label.
  val patchLabel: Label = new Label(currentX + "," + currentY + "," + windowWidth + "," + windowHeight, patchIcon, Alignment.Center) {
    //Set the position of its text, relative to its icon:
    verticalTextPosition = Alignment.Bottom
    horizontalTextPosition = Alignment.Center
  }
//  label1.icon_=(imageIcon)
  
  //Create the other labels.
  val imageLabel = new Label("Main image",imageIcon, Alignment.Center)
  val descriptionLabel = new Label("Testing how to change label text upon button push")
  //Create tool tips, for the heck of it.
  patchLabel.tooltip = "Patch " + currentX + "," + currentY + "," + windowWidth + "," + windowHeight
  val rButton = new Button("Rural")
  val uButton = new Button("Urban")
  val dunnoButton = new Button("Dunno...")
  val predictButton = new Button("Generate Prediction")
  this.listenTo(rButton)
  this.listenTo(uButton)
  this.listenTo(dunnoButton)
  this.listenTo(predictButton)
  this.reactions += {
    case ButtonClicked(`rButton`) => //do something when button is clicked
      descriptionLabel.text_=("You clicked rural")
      updateImage(Classifier.rural)
      newPatch
    case ButtonClicked(`uButton`) => //do something
      descriptionLabel.text_=("You clicked urban")
      updateImage(Classifier.urban)
      newPatch
    case ButtonClicked(`dunnoButton`) =>
      descriptionLabel.text_=("You moron...")
      newPatch
    case ButtonClicked(`predictButton`) =>
      descriptionLabel.text_=("Building the prediction from your labels...  This could take a while.  Go get a drink")
//      runLearner(1)
  }
  
  def newPatch {
    currentX = r.nextInt(image.width - windowWidth)
    currentY = r.nextInt(image.height - windowHeight)
    val newPatchIcon = Swing.Icon(image.subimage(currentX, currentY, windowWidth, windowHeight).scale(3.0, ScaleMethod.Bicubic).toNewBufferedImage(BufferedImage.TYPE_4BYTE_ABGR))
    //set icon, tooltip, description
    patchLabel.icon_=(newPatchIcon)
    patchLabel.tooltip = "Patch " + currentX + "," + currentY + "," + windowWidth + "," + windowHeight
    patchLabel.text_=("Patch " + currentX + "," + currentY + "," + windowWidth + "," + windowHeight)
  }
  
  def updateImage(label: Int) {
    classifier.labeled += ((currentX, currentY) -> label)
    if (classifier.labeled.size > 0) classifier.getClassPrior
    //color main image
    val newImageIcon = Swing.Icon(classifier.labeledImage.scale(0.1, ScaleMethod.Bicubic).toNewBufferedImage(BufferedImage.TYPE_4BYTE_ABGR))
    imageLabel.icon_=(newImageIcon)
  }
  
  def runLearner(learnerIndex: Int) {
    //TODO: use learner index to apply particular algorithm
//    val newImageIcon = Swing.Icon(classifier.predictedImage.scale(0.1, ScaleMethod.Bicubic).toNewBufferedImage(BufferedImage.TYPE_4BYTE_ABGR))
//    imageLabel.icon_=(newImageIcon)
  }
  
//  def refreshImage: sImage = {
    
    //color labeled regions
//    val img: sImage = image.map{(x: Int, y: Int, p: Pixel) => 
//        val label = pixelLabel(x,y,windowWidth, windowHeight, labeled)
//        label match {
//          case -1 => p
//          case Classifier.rural => Pixel(0, p.green * 4, 0, p.alpha) 
//          case Classifier.urban => Pixel(p.red * 4, 0, 0, p.alpha)
//        }
//    	}
//    img
//    image
//  }
  
  //returns class of position - 0 if not labeled yet, otherwise class according to Classifier.rural/ Classifier.urban
//  def pixelLabel(x: Int, y: Int, w: Int, h: Int, map: scala.collection.mutable.Map[(Int,Int),Int]): Int = {
//    val found: Vector[((Int,Int),Int)] = map.filter{xyLabel => x > xyLabel._1._1 && x < (xyLabel._1._1 + w) && y > xyLabel._1._2 && y < (xyLabel._1._2 + h)}.toVector
//    val classLabel = if (found.length > 0) found(0)._2 else -1
//    classLabel
//  }
  
  val leftPanel = new GridPanel(2,1){
    contents += patchLabel
    contents += new GridPanel(2,1){
      contents += new FlowPanel{
        contents += rButton
        contents += uButton
        contents += dunnoButton
        contents += predictButton
      }
      contents += descriptionLabel
    }
  }
  contents += imageLabel
  contents += leftPanel

  
}

object LabelDemo extends SimpleSwingApplication {
  lazy val top = new MainFrame() {
    title = "GMM Active Learner"
    contents = new LearnerGUI()
  }
}












