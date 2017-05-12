// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface VsInterface {
  /** @param state
   * @return value of state */
  Scalar value(Tensor state);

  /** @param state
   * @param value */
  void assign(Tensor state, Scalar value);
}
