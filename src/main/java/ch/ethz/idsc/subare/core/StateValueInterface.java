// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

// TODO determine if interface is useful
public interface StateValueInterface extends ExpectedRewardInterface {
  /** @param state
   * @return */
  Tensor transitions(Tensor state);

  /** @param state
   * @param next
   * @return */
  Scalar transitionProbability(Tensor state, Tensor next);
}
