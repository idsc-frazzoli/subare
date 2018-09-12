// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** n-step temporal difference for estimating V(s)
 * 
 * box on p.154 */
// TODO not tested yet
public class NStepTemporalDifference extends DequeDigestAdapter {
  private final VsInterface vs;
  private final DiscountFunction discountFunction;
  private final LearningRate learningRate;
  private final StateActionCounter sac;

  public NStepTemporalDifference(VsInterface vs, Scalar gamma, LearningRate learningRate, StateActionCounter sac) {
    this.vs = vs;
    discountFunction = DiscountFunction.of(gamma);
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    StepInterface last = deque.getLast();
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    rewards.append(vs.value(last.nextState()));
    // ---
    final StepInterface stepInterface = deque.getFirst(); // first step in queue
    // ---
    Tensor state0 = stepInterface.prevState();
    Scalar value0 = vs.value(state0);
    Scalar alpha = learningRate.alpha(stepInterface, sac);
    Scalar delta = discountFunction.apply(rewards).subtract(value0).multiply(alpha);
    vs.increment(state0, delta);
    sac.digest(stepInterface);
  }
}
