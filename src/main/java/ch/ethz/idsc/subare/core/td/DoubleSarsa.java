// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DiscountFunction;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;

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
public class DoubleSarsa extends DequeDigestAdapter {
  private static final Distribution COINFLIPPING = BernoulliDistribution.of(RealScalar.of(0.5));
  // ---
  private final DiscreteModel discreteModel;
  private final DiscountFunction discountFunction;
  private final SarsaType sarsaType;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final LearningRate learningRate1;
  private final LearningRate learningRate2;
  private Scalar epsilon = null;

  /** @param sarsaType
   * @param discreteModel
   * @param qsa1
   * @param qsa2
   * @param learningRate1
   * @param learningRate2 */
  public DoubleSarsa( //
      SarsaType sarsaType, //
      DiscreteModel discreteModel, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      LearningRate learningRate1, //
      LearningRate learningRate2 //
  ) {
    this.discreteModel = discreteModel;
    discountFunction = DiscountFunction.of(discreteModel.gamma());
    this.sarsaType = sarsaType;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.learningRate1 = learningRate1;
    this.learningRate2 = learningRate2;
  }

  /** @param epsilon
   * @return epsilon-greedy policy with respect to (qsa1 + qsa2) / 2 */
  public Policy getEGreedy() {
    DiscreteQsa avg = DiscreteValueFunctions.average((DiscreteQsa) qsa1, (DiscreteQsa) qsa2);
    return EGreedyPolicy.bestEquiprobable(discreteModel, avg, epsilon);
  }

  /** @return greedy policy with respect to (qsa1 + qsa2) / 2 */
  public Policy getGreedy() {
    DiscreteQsa avg = DiscreteValueFunctions.average((DiscreteQsa) qsa1, (DiscreteQsa) qsa2);
    return GreedyPolicy.bestEquiprobable(discreteModel, avg);
  }

  /** @param epsilon to build an e-greedy policy */
  public void setExplore(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  @Override // from DequeDigest
  public void digest(Deque<StepInterface> deque) {
    // randomly select which qsa to read and write
    boolean flip = RandomVariate.of(COINFLIPPING).equals(RealScalar.ZERO); // flip coin, probability 0.5 each
    QsaInterface Qsa1 = flip ? qsa2 : qsa1; // for selecting actions and updating
    QsaInterface Qsa2 = flip ? qsa1 : qsa2; // for evaluation (of actions provided by Qsa1)
    LearningRate LearningRate1 = flip ? learningRate2 : learningRate1; // for updating
    // ---
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    // TODO test if input LearningRate1 is correct
    Sarsa sarsa = sarsaType.supply(discreteModel, Qsa1, LearningRate1); // not used for learning
    sarsa.setExplore(epsilon);
    rewards.append(sarsa.crossEvaluate(deque.getLast().nextState(), Qsa2));
    // ---
    // the code below is identical to Sarsa
    StepInterface first = deque.getFirst();
    Tensor state0 = first.prevState(); // state-action pair that is being updated in Q
    Tensor action0 = first.action();
    Scalar value0 = Qsa1.value(state0, action0);
    Scalar alpha = LearningRate1.alpha(first);
    Scalar delta = discountFunction.apply(rewards).subtract(value0).multiply(alpha);
    Qsa1.assign(state0, action0, value0.add(delta)); // update Qsa1
    LearningRate1.digest(first); // signal to LearningRate1
  }
}
