// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.7)
 * 
 * box on p.138 */
public class Sarsa extends AbstractTemporalDifference {
  private final StandardModel standardModel;
  final PolicyWrap policyWrap;
  private final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;

  public Sarsa( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      StandardModel standardModel, //
      QsaInterface qsa, Scalar gamma, Scalar alpha) {
    super(episodeSupplier, policyInterface);
    this.standardModel = standardModel;
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
    Tensor action1 = policyWrap.next(state1, standardModel.actions(state1));
    Scalar value0 = qsa.value(state0, action0);
    Scalar value1 = qsa.value(state1, action1);
    qsa.increment(state0, action0, //
        reward.add(value1.multiply(gamma)).subtract(value0).multiply(alpha));
  }
}
