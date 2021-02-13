// code by jph
package ch.ethz.idsc.subare.util;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Min;

/** ArgMax returns the index of the first element in the input sequence, which is equal to the max value. */
public class FairArg implements Serializable {
  /** @param tensor
   * @return
   * @throws Exception if tensor is empty, or a scalar */
  public static FairArg max(Tensor tensor) {
    return new FairArg(Max::of, tensor);
  }

  /** @param tensor
   * @return
   * @throws Exception if tensor is empty, or a scalar */
  public static FairArg min(Tensor tensor) {
    return new FairArg(Min::of, tensor);
  }

  /***************************************************/
  private final List<Integer> list;

  private FairArg(BinaryOperator<Scalar> binaryOperator, Tensor tensor) {
    Tensor value = tensor.stream() //
        .map(Scalar.class::cast) //
        .reduce(binaryOperator).get();
    list = IntStream.range(0, tensor.length()) //
        .filter(index -> tensor.get(index).equals(value)) //
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
