// code by jph
package ch.ethz.idsc.subare.util;

import java.io.Serializable;
import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.sca.Chop;

public class RobustArgMax implements Serializable {
  private final Chop chop;

  public RobustArgMax(Chop chop) {
    this.chop = chop;
  }

  /** @param tensor
   * @return first index that is epsilon close to the maximum
   * @throws exception if tensor is empty */
  public int of(Tensor tensor) {
    Scalar max = tensor.stream().reduce(Max::of).get().Get();
    return IntStream.range(0, tensor.length()) //
        .filter(index -> chop.close(tensor.Get(index), max)) //
        .findFirst() //
        .getAsInt();
  }
}
