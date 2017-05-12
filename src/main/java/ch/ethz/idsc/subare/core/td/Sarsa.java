// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.7)
 * 
 * box on p.138
 * 
 * the Sarsa algorithm was introduced by Rummery and Niranjan 1994 */
public class Sarsa extends AbstractTemporalDifference {
  private final DiscreteModel discreteModel;
  final PolicyWrap policyWrap;
  private final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;

  public Sarsa( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel discreteModel, //
      QsaInterface qsa, Scalar gamma, Scalar alpha) {
    super(episodeSupplier, policyInterface);
    this.discreteModel = discreteModel;
    policyWrap = new PolicyWrap(policyInterface);
    this.qsa = qsa;
    this.gamma = gamma;
    this.alpha = alpha;
  }

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action0 = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    Tensor action1 = policyWrap.next(state1, discreteModel.actions(state1));
    Scalar value0 = qsa.value(state0, action0);
    Scalar value1 = qsa.value(state1, action1);
    Scalar delta = reward.add(value1.multiply(gamma)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
