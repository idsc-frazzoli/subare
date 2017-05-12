// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** deterministic reward */
public interface RewardInterface {
  /** @param state
   * @param action
   * @param next
   * @return reward */
  Scalar reward(Tensor state, Tensor action, Tensor next);
}
