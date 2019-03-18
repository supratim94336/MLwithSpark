package utilities

object UtilFunctions {
  def manOf[T: Manifest](t: T): Manifest[T] = manifest[T]
}
