// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** similar to QLearning except does not use episodes but single steps
 * 
 * similar to {@link ActionValueIteration} but with gauss-seidel updates
 * therefore not parallel()
 * 
 * see box on p.169
 * 
 * algorithm performs poorly when rewards are unevenly distributed */
public class Random1StepTabularQPlanning implements StepDigest, StateActionCounterSupplier {
  public static Random1StepTabularQPlanning of(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRat) {
    return new Random1StepTabularQPlanning(discreteModel, qsa, learningRat, new DiscreteStateActionCounter());
  }
  // ---

  private final DiscreteModel discreteModel;
  private final DiscreteQsa qsa;
  private final Scalar gamma;
  private final LearningRate learningRate;
  private final StateActionCounter sac;

  /** @param discreteModel
   * @param qsa
   * @param learningRate for deterministic tasks, a learning rate of constant == 1 is feasible */
  private Random1StepTabularQPlanning( //
      DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate, StateActionCounter sac) {
    this.discreteModel = discreteModel;
    this.qsa = (DiscreteQsa) qsa;
    this.gamma = discreteModel.gamma();
    this.learningRate = learningRate;
    this.sac = sac;
  }

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    // ---
    Scalar max = discreteModel.actions(state1).stream() //
        // ignore un-encountered state-action pairs, otherwise influenced by initial qsa value
        .filter(action1 -> sac.isEncountered(StateAction.key(state1, action1))) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
    Scalar value0 = qsa.value(state0, action);
    Scalar alpha = learningRate.alpha(stepInterface, sac);
    Scalar value1 = reward.add(gamma.multiply(max));
    // the condition permits "Infinity" as initial qsa value
    if (alpha.equals(RealScalar.ONE))
      qsa.assign(state0, action, value1);
    else {
      Scalar delta = value1.subtract(value0).multiply(alpha);
      qsa.assign(state0, action, value0.add(delta));
    }
    sac.digest(stepInterface);
  }

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
