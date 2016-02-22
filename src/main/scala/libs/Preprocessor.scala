package libs

import scala.util.Random

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import scala.collection.mutable._

trait Preprocessor {
  def convert(name: String, shape: Array[Int]): (Any, Array[Float]) => Unit
}

// The convert method in DefaultPreprocessor is used to convert data extracted
// from a dataframe into an NDArray, which can then be passed into a net. The
// implementation in DefaultPreprocessor is slow and does unnecessary
// allocation. This is designed to be easier to understand, whereas the
// ImageNetPreprocessor is designed to be faster.
class DefaultPreprocessor(schema: StructType) extends Preprocessor {
  def convert(name: String, shape: Array[Int]): (Any, Array[Float]) => Unit = {
    schema(name).dataType match {
      case FloatType => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        NDArray(Array[Float](element.asInstanceOf[Float]), shape).flatCopy(buffer)
      }
      case DoubleType => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        NDArray(Array[Float](element.asInstanceOf[Double].toFloat), shape).flatCopy(buffer)
      }
      case IntegerType => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        NDArray(Array[Float](element.asInstanceOf[Int].toFloat), shape).flatCopy(buffer)
      }
      case LongType => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        NDArray(Array[Float](element.asInstanceOf[Long].toFloat), shape).flatCopy(buffer)
      }
      case BinaryType => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        NDArray(element.asInstanceOf[Array[Byte]].map(e => (e & 0xFF).toFloat), shape).flatCopy(buffer)
      }
      case ArrayType(FloatType, true) => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        element match {
          case element: Array[Float] => NDArray(element.asInstanceOf[Array[Float]], shape).flatCopy(buffer)
          case element: WrappedArray[Float] => NDArray(element.asInstanceOf[WrappedArray[Float]].toArray, shape).flatCopy(buffer)
          case element: ArrayBuffer[Float] => NDArray(element.asInstanceOf[ArrayBuffer[Float]].toArray, shape).flatCopy(buffer)
        }
      }
    }
  }
}

class ImageNetPreprocessor(schema: StructType, meanImage: Array[Float], fullHeight: Int = 256, fullWidth: Int = 256, croppedHeight: Int = 227, croppedWidth: Int = 227) extends Preprocessor {
  def convert(name: String, shape: Array[Int]): (Any, Array[Float]) => Unit = {
    schema(name).dataType match {
      case IntegerType => (element: Any, buffer: Array[Float]) => {
        if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
        NDArray(Array[Float](element.asInstanceOf[Int].toFloat), shape).flatCopy(buffer)
      }
      case BinaryType => {
        if (shape(0) != 3) {
          throw new IllegalArgumentException("Expecting input image to have 3 channels.")
        }
        val tempBuffer = new Array[Float](3 * fullHeight * fullWidth)
        (element: Any, buffer: Array[Float]) => {
          if (buffer.length != shape.product) { throw new Exception("buffer.length and shape.product don't agree, buffer has length " + buffer.length.toString + ", but shape is " + shape.deep.toString) }
          element match {
            case element: Array[Byte] => {
              var index = 0
              while (index < 3 * fullHeight * fullWidth) {
                tempBuffer(index) = (element(index) & 0XFF).toFloat - meanImage(index)
                index += 1
              }
            }
          }
          val heightOffset = Random.nextInt(fullHeight - croppedHeight + 1)
          val widthOffset = Random.nextInt(fullWidth - croppedWidth + 1)
          val lowerIndices = Array[Int](0, heightOffset, widthOffset)
          val upperIndices = Array[Int](shape(0), heightOffset + croppedHeight, widthOffset + croppedWidth)
          NDArray(tempBuffer, Array[Int](shape(0), fullHeight, fullWidth)).subarray(lowerIndices, upperIndices).flatCopy(buffer)
        }
      }
    }
  }
}
