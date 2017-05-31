// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StandardModel extends DiscreteModel {
  /** @param state
   * @param action
   * @param gvalues
   * @return expected value of state-action pair */
  Scalar qsa(Tensor state, Tensor action, VsInterface gvalues);
}
