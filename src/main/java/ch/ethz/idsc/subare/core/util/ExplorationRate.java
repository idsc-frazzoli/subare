// code by jph and fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

@FunctionalInterface
public interface ExplorationRate {
  /** @param state
   * @param stateActionCounter
   * @return exploration rate for given state-action pair */
  Scalar epsilon(Tensor state, StateActionCounter stateActionCounter);
}
