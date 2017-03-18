// code by jph
package ch.ethz.idsc.subare.util;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.ArgMax;

public class FairArg {
  public static final Random random = new Random();

  public static int max(Tensor tensor) {
    final int argmax = ArgMax.of(tensor);
    Scalar max = tensor.Get(argmax);
    List<Integer> list = IntStream.range(0, tensor.length()) //
        .boxed() //
        .filter(i -> tensor.Get(i).equals(max)) //
        .collect(Collectors.toList());
    // ---
    int blub = list.get(random.nextInt(list.size()));
    return blub;
  }
}
