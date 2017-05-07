// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** for estimating value of policy
 * using eq (6.2) on p.128
 * 
 * V(S) = V(S) + alpha * [R + gamma * V(S') - V(S)]
 * 
 * see box on p.128 */
public class TabularTemporalDifference0 extends AbstractTemporalDifference {
  private final VsInterface vs;
  private final Scalar alpha;
  private final Scalar gamma;

  public TabularTemporalDifference0( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      VsInterface vs, Scalar alpha, Scalar gamma //
  ) {
    super(episodeSupplier, policyInterface);
    this.vs = vs;
    this.alpha = alpha;
    this.gamma = gamma;
  }

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    Scalar value0 = vs.value(state0);
    Scalar value1 = vs.value(state1);
    vs.increment(state0, //
        reward.add(value1.multiply(gamma)).subtract(value0).multiply(alpha));
  }
}
