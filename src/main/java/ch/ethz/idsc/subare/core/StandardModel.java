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

  // TODO not sure if this is good name
  Scalar qsa(Tensor state, Tensor action, Tensor gvalues);
}
