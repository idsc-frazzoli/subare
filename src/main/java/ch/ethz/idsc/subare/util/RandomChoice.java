// code by jph
package ch.ethz.idsc.subare.util;

import java.security.SecureRandom;
import java.util.List;
import java.util.Random;

import ch.ethz.idsc.tensor.Tensor;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/RandomChoice.html">RandomChoice</a> */
public enum RandomChoice {
  ;
  private static final Random RANDOM = new SecureRandom();

  /** @param tensor
   * @return random entry of tensor
   * @throws Exception if given tensor is empty */
  @SuppressWarnings("unchecked")
  public static <T extends Tensor> T of(Tensor tensor) {
    return (T) tensor.get(RANDOM.nextInt(tensor.length()));
  }

  /***************************************************/
  /** @param list
   * @param random
   * @return random entry of list
   * @throws Exception if given list is empty */
  public static <T> T of(List<T> list, Random random) {
    return list.get(random.nextInt(list.size()));
  }

  /** @param list
   * @return */
  public static <T> T of(List<T> list) {
    return of(list, RANDOM);
  }
}
