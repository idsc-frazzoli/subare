// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/* package */ interface RewardInterface {
  /** the reward function is not necessarily deterministic
   * 
   * @param state
   * @param action
   * @param next
   * @return reward may vary even for invariant input */
  Scalar reward(Tensor state, Tensor action, Tensor next);
}
