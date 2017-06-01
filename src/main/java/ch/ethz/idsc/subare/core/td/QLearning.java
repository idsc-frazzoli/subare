// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** Q-learning: An off-policy TD control algorithm
 * 
 * eq (6.8)
 * 
 * box on p.140
 * 
 * see also Watkins 1989 */
// TODO check convergence criteria
public class QLearning extends AbstractTemporalDifference {
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;

  /** @param episodeSupplier
   * @param policyInterface
   * @param discreteModel
   * @param qsa
   * @param alpha update rate */
  public QLearning( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha) {
    super(episodeSupplier, policyInterface);
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.gamma = discreteModel.gamma();
    this.alpha = alpha;
  }

  @Override
  public final void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action0 = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    Scalar max = discreteModel.actions(state1).flatten(0) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa.value(state0, action0);
    Scalar delta = reward.add(gamma.multiply(max)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
