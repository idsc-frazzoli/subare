// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** a policy is a function that maps a state-action pair to a real number
 * 
 * the domain is identical to that of a {@link QsaInterface}
 * 
 * since a PolicyInterface outputs probabilities, the additional constraint holds:
 * sum_a pi(a|s) == 1 for all states s */
public interface Policy {
  /** @param state
   * @param action
   * @return probability that action is taken when in given state */
  Scalar probability(Tensor state, Tensor action);
}
