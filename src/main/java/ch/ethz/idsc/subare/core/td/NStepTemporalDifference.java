// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** n-step temporal difference for estimating V(s)
 * 
 * box on p.154 */
// TODO not tested yet
public class NStepTemporalDifference implements DequeDigest {
  private final VsInterface vs;
  private final Scalar gamma;
  private final LearningRate learningRate;

  public NStepTemporalDifference(VsInterface vs, Scalar gamma, LearningRate learningRate) {
    this.vs = vs;
    this.gamma = gamma;
    this.learningRate = learningRate;
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    StepInterface last = deque.getLast();
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    rewards.append(vs.value(last.nextState()));
    // ---
    final StepInterface stepInterface = deque.getFirst(); // first step in queue
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    // ---
    Scalar alpha = learningRate.alpha(state0, action);
    Scalar value0 = vs.value(state0);
    vs.assign(state0, value0.add(Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha)));
    learningRate.digest(stepInterface);
  }
}
