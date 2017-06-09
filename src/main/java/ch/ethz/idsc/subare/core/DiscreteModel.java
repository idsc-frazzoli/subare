// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface DiscreteModel extends DiscountInterface {
  /** @return all states */
  Tensor states();

  /** for a terminal state, the returned actions(state) should have length() == 1
   * 
   * @param state
   * @return all actions possible to execute from given state */
  Tensor actions(Tensor state);
}
