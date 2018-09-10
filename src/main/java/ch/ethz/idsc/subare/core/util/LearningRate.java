// code by jph and fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;

public interface LearningRate {
  /** successive calls to the function give the same result.
   * 
   * the first call to the function should return numerical value == 1
   * to prevent initialization bias.
   * 
   * @param {@link StepInterface}
   * @param {@link StateActionCounter}
   * @return learning rate for given state-action pair */
  Scalar alpha(StepInterface stepInterface, StateActionCounter stateActionCounter);
}
