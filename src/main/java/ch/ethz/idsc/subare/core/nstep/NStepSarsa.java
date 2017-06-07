// code by jph
package ch.ethz.idsc.subare.core.nstep;

import java.util.Deque;
import java.util.Random;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** n-step Sarsa for estimating Q(s,a)
 * 
 * box on p. 157 */
public class NStepSarsa implements DequeDigest {
  private final Random random = new Random();
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;
  private final PolicyInterface policyInterface;

  public NStepSarsa(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha, //
      PolicyInterface policyInterface) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.gamma = discreteModel.gamma();
    this.alpha = alpha;
    this.policyInterface = policyInterface;
  }

  // TODO reuse from original sarsa, expected sarsa, etc
  Scalar evaluate(Tensor state1) {
    PolicyWrap policyWrap = new PolicyWrap(policyInterface, random);
    Tensor action1 = policyWrap.next(state1, discreteModel.actions(state1));
    return qsa.value(state1, action1); // estimate of value of state at the end of deque
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    rewards.append( //
        evaluate(deque.getLast().nextState()) //
    );
    // ---
    StepInterface first = deque.getFirst();
    Tensor state0 = first.prevState();
    Tensor action0 = first.action();
    Scalar value0 = qsa.value(state0, action0);
    Scalar delta = Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
