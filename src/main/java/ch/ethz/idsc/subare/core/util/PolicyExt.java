// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterfaceSupplier;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.tensor.Tensor;

public interface PolicyExt extends Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
  /** @param state
   * @return vector of actions that are equally optimal */
  Tensor getBestActions(Tensor state);

  PolicyExt copy();
}
