// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** class provides the four entries (s,a,r,s')
 * 
 * instances of implementations are required to be immutable
 * Example: {@link StepAdapter} */
public interface StepInterface {
  /** @return previous state */
  Tensor prevState();

  /** @return action that was taken to reach next state */
  Tensor action();

  /** @return reward */
  Scalar reward();

  /** @return next state */
  Tensor nextState();
}
