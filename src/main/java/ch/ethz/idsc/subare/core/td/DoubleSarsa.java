// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;
import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

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
  private final DiscreteModel discreteModel;
  private final SarsaType type;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final Scalar gamma;
  private final LearningRate learningRate;
  private final PolicyInterface policyInterface;
  private final Random random = new Random();

  /** @param type
   * @param discreteModel
   * @param qsa1
   * @param qsa2
   * @param learningRate
   * @param policyInterface */
  public DoubleSarsa( //
      SarsaType type, //
      DiscreteModel discreteModel, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      LearningRate learningRate, //
      PolicyInterface policyInterface //
  ) {
    this.discreteModel = discreteModel;
    this.type = type;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.gamma = discreteModel.gamma();
    this.learningRate = learningRate;
    this.policyInterface = policyInterface;
  }

  public PolicyInterface getEGreedy(Scalar epsilon) {
    DiscreteQsa dqsa1 = (DiscreteQsa) qsa1;
    DiscreteQsa dqsa2 = (DiscreteQsa) qsa2;
    Tensor value = dqsa1.values().add(dqsa2.values());
    return EGreedyPolicy.bestEquiprobable( //
        discreteModel, dqsa1.create(value.flatten(0)), epsilon);
  }

  @Override
  public void digest(Deque<StepInterface> deque) {
    // randomly select which qsa to read and write
    boolean flip = random.nextBoolean(); // flip coin, probability 0.5 each
    QsaInterface Qsa1 = flip ? qsa2 : qsa1;
    QsaInterface Qsa2 = flip ? qsa1 : qsa2;
    // ---
    Tensor rewards = Tensor.of(deque.stream().map(StepInterface::reward));
    // ---
    Tensor stateP = deque.getLast().nextState(); // S' == "state prime"
    switch (type) {
    case original:
    case qlearning: {
      // TODO for original sarsa, the policyInterface is probably wrong!
      ActionSarsa actionSarsa = (ActionSarsa) type.supply(discreteModel, Qsa1, learningRate);
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
    Scalar alpha = learningRate.alpha(first);
    // TODO need to call learning rate digest!
    Scalar delta = Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha);
    Qsa1.assign(state0, action0, value0.add(delta)); // update Qsa1
  }
}
