// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StandardModel extends DiscreteModel {
  /** function implements the formula
   * 
   * sum_(s',r) p(s',r|s,a) [r + gamma * v(s')]
   * 
   * @param state
   * @param action
   * @param gvalues discounted by gamma
   * @return expected value of state-action pair */
  Scalar qsa(Tensor state, Tensor action, VsInterface gvalues);
}
