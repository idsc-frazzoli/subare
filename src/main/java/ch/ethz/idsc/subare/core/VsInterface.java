// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface VsInterface {
  /** @param state
   * @return value of state */
  Scalar value(Tensor state);

  void increment(Tensor state, Scalar delta);

  /** map state to value
   * 
   * @param state
   * @param value */
  void assign(Tensor state, Scalar value);

  /** @return modifiable duplicate of this instance */
  VsInterface copy();

  /** @param gamma
   * @return */
  VsInterface discounted(Scalar gamma);
}
