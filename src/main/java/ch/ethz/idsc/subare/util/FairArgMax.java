// code by jph
package ch.ethz.idsc.subare.util;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

public class FairArgMax implements Serializable {
  /** @param tensor
   * @return
   * @throws Exception if tensor is empty, or a scalar */
  public static FairArgMax of(Tensor tensor) {
    return new FairArgMax(tensor);
  }

  // ---
  private final List<Integer> list;

  private FairArgMax(Tensor tensor) {
    Scalar max = tensor.stream().reduce(Max::of).get().Get();
    list = IntStream.range(0, tensor.length()) //
        .filter(index -> tensor.Get(index).equals(max)) //
        .boxed() //
        .collect(Collectors.toList());
  }

  public int nextRandomIndex() {
    return RandomChoice.of(list);
  }

  public boolean isUnique() {
    return list.size() == 1;
  }

  public int optionsCount() {
    return list.size();
  }

  public List<Integer> options() {
    return Collections.unmodifiableList(list);
  }
}
