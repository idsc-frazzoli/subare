// code by jph and fluric
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** interface to indicate how often the state-action pair is visited during learning */
public interface StateActionCounter {
  /** @param key same as listed in {@link DiscreteQsa#keys()}
   * @return number of updates of qsa value for given state-action pair due to learning */
  Scalar counts(Tensor key);

  /** function exists to remove the initialization bias
   * 
   * @param key that contains the state-action pair
   * @return whether given (state, action) pair has already been encountered by learning rate */
  boolean isEncountered(Tensor key);

  /** @param stepInterface
   * does update the state-action count */
  void digest(StepInterface stepInterface);
}
