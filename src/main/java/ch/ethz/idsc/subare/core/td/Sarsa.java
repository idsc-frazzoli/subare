// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.ExactNumberQ;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** suggested base class for implementations of
 * 
 * - Original Sarsa
 * - Expected Sarsa
 * - Q-Learning
 * 
 * Sarsa provide the capability to digest
 * 
 * - a single step {@link StepDigest},
 * - as well as N-steps {@link DequeDigest} */
public abstract class Sarsa implements StepDigest, DequeDigest {
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
    if (ExactNumberQ.of(alpha)) // TODO printout only once
      System.out.println("warning: symbolic values for alpha slow down the software");
  }

  /** @param state1
   * @return value estimation of state1 */
  protected abstract Scalar evaluate(Tensor state1);

  @Override
  public final void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action0 = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    Scalar value0 = qsa.value(state0, action0);
    // ---
    Scalar value1 = evaluate(state1); // <- call implementation
    // ---
    // [reward value1] . [gamma^0 gamma^1]
    Scalar delta = reward.add(gamma.multiply(value1)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
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
