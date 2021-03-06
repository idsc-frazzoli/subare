// code by jph
package ch.ethz.idsc.subare.util;

import java.util.List;
import java.util.Random;

import ch.ethz.idsc.tensor.Tensor;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/RandomChoice.html">RandomChoice</a> */
public enum RandomChoice {
  ;
  private static final Random RANDOM = new Random();

  /** @param tensor
   * @return random entry of tensor
   * @throws Exception if given tensor is empty */
  @SuppressWarnings("unchecked")
  public static <T extends Tensor> T of(Tensor tensor) {
    return (T) tensor.get(RANDOM.nextInt(tensor.length()));
  }

  /** @param list
   * @return random entry of list
   * @throws Exception if given list is empty */
  public static <T> T of(List<T> list) {
    return list.get(RANDOM.nextInt(list.size()));
  }
}
