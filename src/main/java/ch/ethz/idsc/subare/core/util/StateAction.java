// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public enum StateAction {
  ;
  /** @param state
   * @param action
   * @return */
  public static Tensor key(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  /** @param stepInterface
   * @return */
  public static Tensor key(StepInterface stepInterface) {
    return key(stepInterface.prevState(), stepInterface.action());
  }
}
