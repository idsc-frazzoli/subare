// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface ActionValueInterface extends ExpectedRewardInterface {
  /** @param state
   * @param action
   * @return all states that are a possible result of taking action in given state */
  Tensor transitions(Tensor state, Tensor action);

  /** @param state
   * @param action
   * @param next
   * @return probability to reach next as a result of taking action in given state */
  Scalar transitionProbability(Tensor state, Tensor action, Tensor next);
}
