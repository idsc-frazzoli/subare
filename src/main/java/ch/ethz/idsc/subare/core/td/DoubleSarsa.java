// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.subare.core.util.StateActionCounterUtil;
import ch.ethz.idsc.subare.util.Coinflip;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** double sarsa for single-step, and n-step
 * 
 * implementation covers
 * 
 * Double Q-learning (box on p.145)
 * Double Expected Sarsa (Exercise 6.10 p.145)
 * Double Original Sarsa (p.145)
 * 
 * the update equation in the box uses argmax_a.
 * since there may not be a unique best action, we average the evaluation
 * over all best actions.
 * 
 * Maximization bias and Doubled learning were introduced and investigated
 * by Hado van Hasselt (2010, 2011) */
public class DoubleSarsa extends DequeDigestAdapter implements DiscreteQsaSupplier, StateActionCounterSupplier {
  private final Coinflip coinflip = Coinflip.fair();
  // ---
  private final DiscreteModel discreteModel;
  private final DiscountFunction discountFunction;
  private final SarsaEvaluation sarsaEvaluation;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final LearningRate learningRate1;
  private final LearningRate learningRate2;
  private final StateActionCounter sac1;
  private final StateActionCounter sac2;
  private Scalar epsilon = null;

  /** @param sarsaEvaluation
   * @param discreteModel
   * @param learningRate1
   * @param learningRate2
   * @param qsa1
   * @param qsa2
   * @param sac1
   * @param sac2 */
  /* package */ DoubleSarsa( //
      SarsaEvaluation sarsaEvaluation, //
      DiscreteModel discreteModel, //
      LearningRate learningRate1, //
      LearningRate learningRate2, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      StateActionCounter sac1, //
      StateActionCounter sac2 //
  ) {
    this.discreteModel = discreteModel;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.sarsaEvaluation = sarsaEvaluation;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.learningRate1 = learningRate1;
    this.learningRate2 = learningRate2;
    this.sac1 = sac1;
    this.sac2 = sac2;
  }

  /** @param epsilon
   * @return epsilon-greedy policy with respect to (qsa1 + qsa2) / 2 */
  public Policy getEGreedy() {
    DiscreteQsa avg = DiscreteValueFunctions.average((DiscreteQsa) qsa1, (DiscreteQsa) qsa2);
    return new EGreedyPolicy(discreteModel, avg, epsilon);
  }

  /** @return greedy policy with respect to (qsa1 + qsa2) / 2 */
  public Policy getGreedy() {
    DiscreteQsa avg = DiscreteValueFunctions.average((DiscreteQsa) qsa1, (DiscreteQsa) qsa2);
    return GreedyPolicy.of(discreteModel, avg);
  }

  /** @param epsilon to build an e-greedy policy */
  public void setExplore(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  @Override // from DequeDigest
  public void digest(Deque<StepInterface> deque) {
    // randomly select which qsa to read and write
    boolean flip = coinflip.tossHead(); // flip coin, probability 0.5 each
    QsaInterface Qsa1 = flip ? qsa2 : qsa1; // for selecting actions and updating
    QsaInterface Qsa2 = flip ? qsa1 : qsa2; // for evaluation (of actions provided by Qsa1)
    LearningRate LearningRate = flip ? learningRate2 : learningRate1; // for updating
    StateActionCounter Sac = flip ? sac2 : sac1; // for updating
    // ---
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    // TODO test if input LearningRate1 is correct
    Tensor nextState = deque.getLast().nextState();
    Tensor nextActions = Tensor.of(discreteModel.actions(nextState).stream() //
        .filter(nextAction -> Sac.isEncountered(StateAction.key(nextState, nextAction))));
    Scalar expectedReward = Tensors.isEmpty(nextActions) //
        ? RealScalar.ZERO
        : sarsaEvaluation.crossEvaluate(epsilon, nextState, nextActions, Qsa1, Qsa2);
    rewards.append(expectedReward);
    // ---
    // the code below is identical to Sarsa
    StepInterface first = deque.getFirst();
    Tensor state0 = first.prevState(); // state-action pair that is being updated in Q
    Tensor action0 = first.action();
    Scalar value0 = Qsa1.value(state0, action0);
    Scalar alpha = LearningRate.alpha(first, Sac);
    Scalar delta = discountFunction.apply(rewards).subtract(value0).multiply(alpha);
    Qsa1.assign(state0, action0, value0.add(delta)); // update Qsa1
    Sac.digest(first); // signal to LearningRate1
  }

  @Override
  public DiscreteQsa qsa() {
    return DiscreteValueFunctions.weightedAverage( //
        (DiscreteQsa) qsa1, (DiscreteQsa) qsa2, sac1, sac2);
  }

  @Override
  public StateActionCounter sac() {
    return StateActionCounterUtil.getSummedSac(sac1, sac2, discreteModel);
  }
}
