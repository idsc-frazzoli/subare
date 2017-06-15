// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;
import ch.ethz.idsc.tensor.pdf.BernoulliDistribution;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;

/** double sarsa for single-step, and n-step
 * 
 * implementation covers
 * 
 * Double Q-learning (box on p.145)
 * Double Expected Sarsa (Exercise 6.10)
 * Double Original Sarsa (p.145)
 * 
 * Maximization bias and Doubled learning were introduced and investigated
 * by Hado van Hasselt (2010, 2011) */
public class DoubleSarsa extends DequeDigestAdapter {
  private static final Distribution COINFLIPPING = BernoulliDistribution.of(RationalScalar.of(1, 2));
  // ---
  private final DiscreteModel discreteModel;
  private final SarsaType sarsaType;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final Scalar gamma;
  private final LearningRate learningRate1;
  private final LearningRate learningRate2;
  private PolicyInterface policyInterface = null;

  /** @param sarsaType
   * @param discreteModel
   * @param qsa1
   * @param qsa2
   * @param learningRate1
   * @param learningRate2
   * @param policyInterface */
  public DoubleSarsa( //
      SarsaType sarsaType, //
      DiscreteModel discreteModel, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      LearningRate learningRate1, //
      LearningRate learningRate2 //
  ) {
    this.discreteModel = discreteModel;
    this.sarsaType = sarsaType;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.gamma = discreteModel.gamma();
    this.learningRate1 = learningRate1;
    this.learningRate2 = learningRate2;
  }

  public PolicyInterface getEGreedy(Scalar epsilon) {
    DiscreteQsa avg = TensorValuesUtils.average((DiscreteQsa) qsa1, (DiscreteQsa) qsa2);
    return EGreedyPolicy.bestEquiprobable(discreteModel, avg, epsilon);
  }

  /** @param policyInterface that is used to generate the {@link StepInterface} */
  public void setPolicyInterface(PolicyInterface policyInterface) {
    this.policyInterface = policyInterface;
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    // randomly select which qsa to read and write
    boolean flip = RandomVariate.of(COINFLIPPING).equals(RealScalar.ZERO); // flip coin, probability 0.5 each
    QsaInterface Qsa1 = flip ? qsa2 : qsa1; // for updating
    QsaInterface Qsa2 = flip ? qsa1 : qsa2; // for evaluation
    LearningRate LearningRate1 = flip ? learningRate2 : learningRate1;
    // ---
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    // ---
    Tensor stateP = deque.getLast().nextState(); // S' == "state prime"
    switch (sarsaType) {
    case original:
    case qlearning: {
      // TODO for original sarsa, the policyInterface is probably wrong!
      ActionSarsa actionSarsa = (ActionSarsa) sarsaType.supply(discreteModel, Qsa1, LearningRate1);
      actionSarsa.setPolicyInterface(policyInterface);
      Tensor action = actionSarsa.actionForEvaluation(stateP); // use Qsa1 to select action
      rewards.append(Qsa2.value(stateP, action)); // use Qsa2 to evaluate state-action pair
      break;
    }
    case expected:
      // TODO figure out formula for double expected sarsa
    default:
      throw new RuntimeException();
    }
    // ---
    // the code below is identical to Sarsa
    StepInterface first = deque.getFirst();
    Tensor state0 = first.prevState(); // state-action pair that is being updated in Q
    Tensor action0 = first.action();
    Scalar value0 = Qsa1.value(state0, action0);
    Scalar alpha = LearningRate1.alpha(first);
    Scalar delta = Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha);
    Qsa1.assign(state0, action0, value0.add(delta)); // update Qsa1
    LearningRate1.digest(first); // signal to LearningRate1
  }
}
