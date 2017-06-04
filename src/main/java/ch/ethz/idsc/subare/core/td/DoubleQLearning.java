// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** Double Q-learning
 * 
 * box on p.145 */
public class DoubleQLearning implements StepDigest {
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa1;
  private final QsaInterface qsa2;
  private final Scalar gamma;
  private final Scalar alpha;
  private final Random random = new Random();

  /** @param discreteModel
   * @param qsa1
   * @param qsa2
   * @param alpha update rate */
  public DoubleQLearning( //
      DiscreteModel discreteModel, //
      QsaInterface qsa1, //
      QsaInterface qsa2, //
      Scalar alpha) {
    this.discreteModel = discreteModel;
    this.qsa1 = qsa1;
    this.qsa2 = qsa2;
    this.gamma = discreteModel.gamma();
    this.alpha = alpha;
  }

  public PolicyInterface getEGreedy(Scalar epsilon) {
    DiscreteQsa dqsa1 = (DiscreteQsa) qsa1;
    DiscreteQsa dqsa2 = (DiscreteQsa) qsa2;
    Tensor value = dqsa1.values().add(dqsa2.values());
    return EGreedyPolicy.bestEquiprobable(discreteModel, dqsa1.create(value.flatten(0)), epsilon);
  }

  @Override
  public final void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action0 = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    // TODO
    // random.nextBoolean()
    Scalar max = discreteModel.actions(state1).flatten(0) //
        .map(action1 -> qsa1.value(state1, action1)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa1.value(state0, action0);
    Scalar delta = reward.add(gamma.multiply(max)).subtract(value0).multiply(alpha);
    qsa1.assign(state0, action0, value0.add(delta));
  }
}
