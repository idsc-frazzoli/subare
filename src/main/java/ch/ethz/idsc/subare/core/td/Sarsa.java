// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

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
 * https://github.com/idsc-frazzoli/subare/files/2257711/sarsa.pdf
 * 
 * the abstract class Sarsa provides the capability to digest:
 * 
 * a single step {@link StepDigest},
 * as well as N-steps {@link DequeDigest} */
public class Sarsa extends DequeDigestAdapter implements DiscreteQsaSupplier {
  private final SarsaEvaluation sarsaEvaluation;
  private final DiscountFunction discountFunction;
  private final QsaInterface qsa;
  private final LearningRate learningRate;
  // ---
  private Scalar epsilon = null;

  /** @param discreteModel
   * @param qsa
   * @param learningRate
   * @param sarsaEvaluation */
  /* package */ Sarsa(SarsaEvaluation sarsaEvaluation, DiscreteModel discreteModel, LearningRate learningRate, QsaInterface qsa) {
    this.sarsaEvaluation = sarsaEvaluation;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.qsa = qsa;
    this.learningRate = learningRate;
  }

  /** @param epsilon */
  public final void setExplore(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  @Override // from DequeDigest
  public final void digest(Deque<StepInterface> deque) {
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    Tensor nextState = deque.getLast().nextState();
    // ---
    // for terminal state in queue, "=last.next"
    rewards.append(sarsaEvaluation.evaluate(epsilon, learningRate, nextState, qsa)); // <- evaluate(...) is called here
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
    Tensor rewards = Tensors.of(stepInterface.reward(), sarsaEvaluation.evaluate(epsilon, learningRate, stepInterface.nextState(), qsa));
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
