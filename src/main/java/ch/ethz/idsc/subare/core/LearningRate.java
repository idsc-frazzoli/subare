// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** functionality to implement formula's for learning rate
 * that may depend on the {@link StepInterface} seen in the episodes */
public interface LearningRate extends StepDigest {
  /** successive calls to the function give the same result.
   * 
   * the first call to the function should return numerical value == 1
   * to prevent initialization bias.
   * 
   * the learning rate may chance only upon calling {@link StepDigest#digest}.
   * 
   * @param state
   * @param action
   * @return learning rate for given state-action pair */
  Scalar alpha(StepInterface stepInterface);

  /** function exists to remove the initialization bias
   * 
   * @param state
   * @param action
   * @return whether given (state, action) pair has already been encountered by learning rate */
  boolean isEncountered(Tensor state, Tensor action);
}
