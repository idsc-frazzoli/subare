// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

// not yet used
public interface RewardInterface {
  /** @param state
   * @param action
   * @param next
   * @return */
  Scalar reward(Tensor state, Tensor action, Tensor next);
}
