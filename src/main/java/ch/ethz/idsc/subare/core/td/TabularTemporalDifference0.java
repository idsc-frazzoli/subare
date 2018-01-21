// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Tabular TD(0) for estimating value of policy
 * using eq (6.2) on p.95
 * <pre>
 * V(S) = V(S) + alpha * [R + gamma * V(S') - V(S)]
 * </pre>
 * 
 * Implementation also covers
 * Semi-gradient TD(0) for estimating an approximate value function
 * in 9.3, p. 164 */
public class TabularTemporalDifference0 implements StepDigest {
  private final VsInterface vs;
  private final Scalar gamma;
  private final LearningRate learningRate;

  /** @param vs
   * @param gamma discount factor
   * @param learningRate */
  public TabularTemporalDifference0( //
      VsInterface vs, Scalar gamma, LearningRate learningRate) {
    this.vs = vs;
    this.gamma = gamma;
    this.learningRate = learningRate;
  }

  @Override // from StepDigest
  public final void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    // action is only required for learning rate
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    // ---
    Scalar value0 = vs.value(state0);
    Scalar value1 = vs.value(state1);
    Scalar alpha = learningRate.alpha(stepInterface);
    Scalar delta = reward.add(gamma.multiply(value1)).subtract(value0).multiply(alpha);
    vs.increment(state0, delta);
    learningRate.digest(stepInterface);
  }
}
