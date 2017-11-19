// code by jph
package ch.ethz.idsc.subare.util;

import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.ArgMax;
import ch.ethz.idsc.tensor.sca.Chop;

public class RobustArgMax {
  private final Chop chop;

  public RobustArgMax(Chop chop) {
    this.chop = chop;
  }

  /** @param tensor
   * @return first index that is epsilon close to the maximum
   * @throws exception if tensor is empty */
  public int of(Tensor tensor) {
    final int argmax = ArgMax.of(tensor);
    Scalar max = tensor.Get(argmax);
    return IntStream.range(0, tensor.length()) //
        .boxed() //
        .filter(index -> chop.close(tensor.Get(index), max)) //
        .findFirst().get();
  }
}
