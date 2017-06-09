// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Deque;
import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DequeDigestAdapter;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** double sarsa for single-step, and n-step
 * 
 * algorithm covers
 * 
 * Double Q-learning (box on p.145)
 * Double Expected Sarsa (Exercise 6.10)
 * Double Original Sarsa (p.145) */
public class DoubleSarsa extends DequeDigestAdapter {
  private final DiscreteModel discreteModel;
  private final SarsaType type;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final Scalar gamma;
  private final Scalar alpha;
  private final PolicyInterface policyInterface;
  private final Random random = new Random();

  /** @param type
   * @param discreteModel
   * @param qsa1
   * @param qsa2
   * @param alpha
   * @param policyInterface */
  public DoubleSarsa( //
      SarsaType type, //
      DiscreteModel discreteModel, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      Scalar alpha, //
      PolicyInterface policyInterface //
  ) {
    this.discreteModel = discreteModel;
    this.type = type;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.gamma = discreteModel.gamma();
    this.alpha = alpha;
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
    // TODO notation "stateP" is old...
    Tensor stateP = deque.getLast().nextState(); // S'
    Sarsa sarsa = type.supply(discreteModel, Qsa1, alpha, policyInterface);
    // TODO implementation is fine for QLearning and original sarsa but NOT for esarsa!
    Tensor actionP = sarsa.evaluate(stateP); // FIXME this is now REALLY WRONG action != value
    rewards.append(Qsa2.value(stateP, actionP));
    // ---
    StepInterface first = deque.getFirst();
    Tensor state0 = first.prevState();
    Tensor action0 = first.action();
    // ---
    Scalar value0 = Qsa1.value(state0, action0);
    Scalar delta = Multinomial.horner(rewards, gamma).subtract(value0).multiply(alpha);
    Qsa2.assign(state0, action0, value0.add(delta));
  }
}
