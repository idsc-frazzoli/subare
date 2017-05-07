// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface VsInterface {
  /** @param state
   * @param action
   * @return value of state action pair */
  Scalar value(Tensor state);

  /** @param state
   * @param action
   * @param delta amount that value of given state action pair should be changed */
  void increment(Tensor state, Scalar delta);
}
