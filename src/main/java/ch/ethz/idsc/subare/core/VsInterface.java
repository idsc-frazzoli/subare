// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** function that maps a given state to a value
 * in addition, the interface provides update */
public interface VsInterface {
  /** @param state
   * @return value of state */
  Scalar value(Tensor state);

  /** update value function to account for delta at given state
   * 
   * @param state
   * @param delta */
  void increment(Tensor state, Scalar delta);

  /** @return modifiable duplicate of this instance */
  VsInterface copy();

  /** @param gamma
   * @return */
  VsInterface discounted(Scalar gamma);
}
