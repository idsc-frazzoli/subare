// code by jph
package ch.ethz.idsc.subare.util;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.ArgMax;

public class FairArgMax {
  private static final Random random = new Random();

  /** @param tensor
   * @return
   * @throws exception if tensor is empty */
  public static FairArgMax of(Tensor tensor) {
    return new FairArgMax(tensor);
  }

  private final List<Integer> list;

  private FairArgMax(Tensor tensor) {
    final int argmax = ArgMax.of(tensor);
    Scalar max = tensor.Get(argmax);
    list = IntStream.range(0, tensor.length()) //
        .boxed() //
        .filter(index -> tensor.Get(index).equals(max)) //
        .collect(Collectors.toList());
  }

  public int nextRandomIndex() {
    return nextRandomIndex(random);
  }

  public int nextRandomIndex(Random random) {
    return list.get(random.nextInt(list.size()));
  }

  public boolean isUnique() {
    return list.size() == 1;
  }

  public int getOptionCount() {
    return list.size();
  }

  public List<Integer> options() {
    return Collections.unmodifiableList(list);
  }
}
