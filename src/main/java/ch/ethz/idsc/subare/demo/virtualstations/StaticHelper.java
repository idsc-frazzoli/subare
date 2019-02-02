// code by fluric
package ch.ethz.idsc.subare.demo.virtualstations;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Join;

enum StaticHelper {
  ;
  /** returns the tensor of all possible binary combinations in a vector of size length
   * 
   * @param length
   * @param prefixes
   * @return */
  static Tensor binaryVectors(int length, Tensor prefixes) {
    if (length == 0)
      return prefixes;
    if (Tensors.isEmpty(prefixes))
      return binaryVectors(length - 1, Tensors.of(Tensors.vector(1), Tensors.vector(0)));
    Tensor extension = Tensors.empty();
    for (Tensor prefix : prefixes) {
      extension.append(Join.of(prefix, Tensors.vector(1)));
      extension.append(Join.of(prefix, Tensors.vector(0)));
    }
    return binaryVectors(length - 1, extension);
  }

  static Tensor zeroVectors(int length, Tensor prefixes) {
    if (Tensors.isEmpty(prefixes))
      return Tensors.of(Array.zeros(length));
    return Tensor.of(prefixes.stream().map(v -> Join.of(v, Array.zeros(length))));
  }
}
