// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface QsaInterface {
  /** @param state
   * @param action
   * @return value of state action pair */
  Scalar value(Tensor state, Tensor action);

  /** @param state
   * @param action
   * @param delta amount that value of given state action pair should be changed */
  void increment(Tensor state, Tensor action, Scalar delta);

  /** @param state
   * @param action
   * @param value */
  void set(Tensor state, Tensor action, Scalar value); // TODO consider removing increment
}
