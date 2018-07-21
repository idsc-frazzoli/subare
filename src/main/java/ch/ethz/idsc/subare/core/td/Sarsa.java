// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;
import java.util.Objects;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

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
public abstract class Sarsa extends DequeDigestAdapter implements DiscreteQsaSupplier {
  private final DiscountFunction discountFunction;
  final DiscreteModel discreteModel;
  final QsaInterface qsa;
  final LearningRate learningRate;
  // ---
  Scalar epsilon = null;

  /** @param discreteModel
   * @param qsa
   * @param learningRate */
  public Sarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    this.discreteModel = discreteModel;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.qsa = Objects.isNull(qsa) ? DiscreteQsa.build(discreteModel) : qsa;
    this.learningRate = learningRate;
  }

  /** @param epsilon */
  public final void setExplore(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  /** @param state
   * @return value estimation of state */
  abstract Scalar evaluate(Tensor state);

  /** @param state
   * @param Qsa2
   * @return value from evaluations of Qsa2 via actions provided by qsa (== Qsa1) */
  final Scalar crossEvaluate(Tensor state, QsaInterface Qsa2) {
    Tensor actions = Tensor.of( //
        discreteModel.actions(state).stream() //
            .filter(action -> learningRate.encountered(state, action)));
    return actions.length() == 0 ? RealScalar.ZERO : crossEvaluate(state, actions, Qsa2);
  }

  /** @param state
   * @param actions non-empty
   * @param Qsa2
   * @return value from evaluations of Qsa2 via actions provided by qsa (== Qsa1) */
  abstract Scalar crossEvaluate(Tensor state, Tensor actions, QsaInterface Qsa2);

  @Override // from DequeDigest
  public final void digest(Deque<StepInterface> deque) {
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
    Scalar alpha = learningRate.alpha(stepInterface);
    Scalar value1 = discountFunction.apply(rewards);
    if (alpha.equals(RealScalar.ONE))
      qsa.assign(state0, action, value1);
    else {
      Scalar delta = value1.subtract(value0).multiply(alpha);
      qsa.assign(state0, action, value0.add(delta));
    }
    // ---
    // since qsa was update for the state-action pair
    // the learning rate interface as well as the usb policy are notified about the state-action pair
    learningRate.digest(stepInterface);
  }

  /** @param stepInterface
   * @return non-negative priority rating */
  final Scalar priority(StepInterface stepInterface) {
    Tensor rewards = Tensors.of(stepInterface.reward(), evaluate(stepInterface.nextState()));
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Scalar value0 = qsa.value(state0, action);
    return discountFunction.apply(rewards).subtract(value0).abs();
  }

  @Override // from DiscreteQsaSupplier
  public final DiscreteQsa qsa() {
    return (DiscreteQsa) qsa;
  }
}
