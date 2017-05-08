// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface PolicyInterface {
  /** @param state
   * @param action
   * @return probability that action is taken when in given state */
  Scalar policy(Tensor state, Tensor action);
}
