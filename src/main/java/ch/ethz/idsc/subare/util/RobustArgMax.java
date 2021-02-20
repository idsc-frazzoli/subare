// code by jph
package ch.ethz.idsc.subare.util;

import java.io.Serializable;
import java.util.stream.IntStream;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.sca.Chop;

/** RobustArgMax accounts for entries that are numerically close to the maximum and
 * returns the first such close match. */
public class RobustArgMax implements Serializable {
  private final Chop chop;

  /** @param chop that performs proximity check to the max via {@link Chop#isClose(Tensor, Tensor)} */
  public RobustArgMax(Chop chop) {
    this.chop = chop;
  }

  /** @param vector
   * @return indices of entries that are close to the maximum entry in vector
   * @throws Exception if vector is empty, or not a tensor of rank 1 */
  public IntStream options(Tensor vector) {
    Tensor max = vector.stream().reduce(Max::of).get();
    return IntStream.range(0, vector.length()) //
        .filter(index -> chop.isClose(vector.get(index), max));
  }

  /** in the spirit of ArgMax which returns the first of equally maximal indices.
   * 
   * @param vector
   * @return first index that is epsilon close to the maximum
   * @throws Exception if vector is empty, or not a tensor of rank 1 */
  public int of(Tensor vector) {
    return options(vector).findFirst().getAsInt();
  }
}
