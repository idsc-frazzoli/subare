// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.pdf.Distribution;

/** a policy is a function that maps a state-action pair to a real number,
 * i.e. the probability that a given action is taken when in given state.
 * 
 * <p>The domain is identical to that of a {@link QsaInterface}
 * 
 * <p>Since a PolicyInterface outputs probabilities, the additional constraint holds:
 * sum_a pi(a|s) == 1 for all states s */
public interface Policy {
  /** Hint: for bulk calls use {@link #getDistribution(Tensor)} because
   * function {@link #probability(Tensor, Tensor)} may not scale.
   * 
   * @param state
   * @param action
   * @return probability that action is taken when in given state,
   * the probability is in the interval [0, 1], and the sum of
   * probabilities of all actions for a given state has to equal 1 */
  Scalar probability(Tensor state, Tensor action);

  /** @param state
   * @return distribution over all possible actions for given state */
  Distribution getDistribution(Tensor state);
}
