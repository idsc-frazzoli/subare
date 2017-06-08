// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DequeDigestAdapter;
import ch.ethz.idsc.tensor.ExactNumberQ;
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
  private final Scalar gamma;
  private final Scalar alpha;

  /** @param discreteModel
   * @param qsa
   * @param alpha learning rate */
  public Sarsa(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.gamma = discreteModel.gamma();
    this.alpha = alpha;
    if (ExactNumberQ.of(alpha)) // TODO printout warning only once
      System.out.println("warning: symbolic values for alpha slow down the software");
  }

  /** @param state
   * @return value estimation of state */
  protected abstract Scalar evaluate(Tensor state);

  @Override
  public void digest(Deque<StepInterface> deque) {
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    // ---
    rewards.append(evaluate(deque.getLast().nextState())); // <- evaluate(...) is called here
    // ---
    StepInterface first = deque.getFirst();
    Tensor state0 = first.prevState();
    Tensor action0 = first.action();
    Scalar value0 = qsa.value(state0, action0);
    Scalar delta = Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
