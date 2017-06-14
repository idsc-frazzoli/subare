// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.UcbPolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** base class for implementations of
 * 
 * {@link OriginalSarsa}
 * {@link ExpectedSarsa}
 * {@link QLearning}
 * 
 * the abstract class Sarsa provides the capability to digest:
 * 
 * a single step {@link StepDigest},
 * as well as N-steps {@link DequeDigest} */
public abstract class Sarsa extends DequeDigestAdapter {
  final DiscreteModel discreteModel;
  final QsaInterface qsa;
  private final LearningRate learningRate;
  PolicyInterface policyInterface = null; // FIXME how does this related to ucb Policy!
  // ---
  private final UcbPolicy ucbPolicy;

  /** @param discreteModel
   * @param qsa
   * @param alpha learning rate */
  public Sarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.learningRate = learningRate;
    ucbPolicy = UcbPolicy.of(discreteModel, qsa, RealScalar.ONE);
  }

  /** @param policyInterface */
  @Deprecated // TODO deprecated only preliminary
  public void setPolicyInterface(PolicyInterface policyInterface) {
    this.policyInterface = policyInterface;
  }

  /** @param policyInterface */
  public UcbPolicy getUcbPolicy() {
    return ucbPolicy;
  }

  /** @param state
   * @return value estimation of state */
  protected abstract Scalar evaluate(Tensor state);

  @Override
  public void digest(Deque<StepInterface> deque) {
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    // ---
    // for terminal state in queue, "=last.next", the sarsa implementation has to provide the evaluation
    rewards.append(evaluate(deque.getLast().nextState())); // <- evaluate(...) is called here
    // ---
    final StepInterface stepInterface = deque.getFirst(); // first step in queue
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    // ---
    Scalar value0 = qsa.value(state0, action);
    Scalar gamma = discreteModel.gamma();
    Scalar alpha = learningRate.alpha(state0, action);
    Scalar delta = Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha);
    qsa.assign(state0, action, value0.add(delta));
    // ---
    // since qsa was update for the state-action pair
    // the learning rate interface as well as the usb policy are notified about the state-action pair
    learningRate.digest(stepInterface);
    ucbPolicy.digest(stepInterface);
  }
}
