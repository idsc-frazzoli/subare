// code by jph
package ch.ethz.idsc.subare.util;

import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.ArgMax;

public enum RobustArgMax {
  ;
  private static final Scalar THRESHOLD = RealScalar.of(1e-6);

  /** @param tensor
   * @return first index that is epsilon close to the maximum
   * @throws exception if tensor is empty */
  public static int of(Tensor tensor) {
    final int argmax = ArgMax.of(tensor);
    Scalar max = tensor.Get(argmax);
    return IntStream.range(0, tensor.length()) //
        .boxed() //
        .filter(index -> Scalars.lessThan(tensor.Get(index).subtract(max).abs(), THRESHOLD)) //
        .findFirst().get();
  }
}
