// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DequeDigest;
import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyExt;
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
public class Sarsa extends DequeDigestAdapter implements DiscreteQsaSupplier, StateActionCounterSupplier {
  private final SarsaEvaluation sarsaEvaluation;
  private final DiscountFunction discountFunction;
  private final QsaInterface qsa;
  private final StateActionCounter sac;
  private final LearningRate learningRate;
  private final PolicyExt policy;

  /** @param sarsaEvaluation
   * @param discreteModel
   * @param learningRate
   * @param qsa
   * @param sac
   * @param policy */
  public Sarsa( //
      SarsaEvaluation sarsaEvaluation, //
      DiscreteModel discreteModel, //
      LearningRate learningRate, //
      QsaInterface qsa, //
      StateActionCounter sac, //
      PolicyExt policy) {
    this.sarsaEvaluation = sarsaEvaluation;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.sac = sac;
    this.qsa = qsa;
    this.learningRate = learningRate;
    this.policy = policy;
  }

  @Override // from DequeDigest
  public final void digest(Deque<StepInterface> deque) {
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    Tensor nextState = deque.getLast().nextState();
    // ---
    // for terminal state in queue, "=last.next"
    // ---
    final StepInterface stepInterface = deque.getFirst(); // first step in queue
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    // ---
    Scalar value0 = qsa.value(state0, action);
    Scalar alpha = learningRate.alpha(stepInterface, sac);
    rewards.append(sarsaEvaluation.evaluate(nextState, policy)); // <- evaluate(...) is called here
    Scalar value1 = discountFunction.apply(rewards);
    if (alpha.equals(RealScalar.ONE))
      qsa.assign(state0, action, value1);
    else {
      Scalar delta = value1.subtract(value0).multiply(alpha);
      qsa.assign(state0, action, value0.add(delta));
    }
    sac.digest(stepInterface);
  }

  /** @param stepInterface
   * @return non-negative priority rating */
  final Scalar priority(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Scalar value0 = qsa.value(state0, action);
    Tensor rewards = Tensors.of(stepInterface.reward(), sarsaEvaluation.evaluate(stepInterface.nextState(), policy));
    return discountFunction.apply(rewards).subtract(value0).abs();
  }

  @Override // from DiscreteQsaSupplier
  public final DiscreteQsa qsa() {
    return (DiscreteQsa) qsa;
  }

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
