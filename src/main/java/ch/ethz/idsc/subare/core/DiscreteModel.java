// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface DiscreteModel {
  /** @return all states */
  Tensor states();

  /** @param state
   * @return all action possible to execute from given state */
  Tensor actions(Tensor state);
}
