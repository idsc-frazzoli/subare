// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** n-step temporal difference for estimating V(s)
 * 
 * box on p. 154 */
// TODO not tested yet
public class NStepTemporalDifference implements DequeDigest {
  private final VsInterface vs;
  private final Scalar gamma;
  private final Scalar alpha;

  public NStepTemporalDifference(VsInterface vs, Scalar gamma, Scalar alpha) {
    this.vs = vs;
    this.gamma = gamma;
    this.alpha = alpha;
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    StepInterface last = deque.getLast();
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    rewards.append(vs.value(last.nextState()));
    Scalar G = Multinomial.horner(rewards, gamma);
    StepInterface first = deque.getFirst();
    Scalar value = vs.value(first.prevState());
    vs.assign(first.prevState(), value.add(G.subtract(value).multiply(alpha)));
  }
}
