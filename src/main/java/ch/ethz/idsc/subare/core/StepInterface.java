// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StepInterface {
  /** @return action that was taken to reach next state */
  Tensor action();

  /** @return reward */
  Scalar reward();

  /** @return next state */
  Tensor nextState();
}
