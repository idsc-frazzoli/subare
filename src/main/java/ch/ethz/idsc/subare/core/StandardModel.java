// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StandardModel {
  /** @return all states */
  Tensor states();

  /** @param state
   * @return all action possible to execute from given state */
  Tensor actions(Tensor state);

  /** @param state
   * @param action
   * @param gvalues
   * @return expected value of state-action pair */
  Scalar qsa(Tensor state, Tensor action, Tensor gvalues);
}
