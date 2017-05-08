// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Expected Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.9) on p.142 */
public class ExpectedSarsa extends AbstractTemporalDifference {
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;

  /** @param episodeSupplier
   * @param policyInterface
   * @param discreteModel
   * @param qsa
   * @param gamma
   * @param alpha if all state transtions are deterministic and all randomness comes
   * from the policy then alpha can be set to 1 */
  public ExpectedSarsa( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel discreteModel, //
      QsaInterface qsa, Scalar gamma, Scalar alpha) {
    super(episodeSupplier, policyInterface);
    this.discreteModel = discreteModel;
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
    // ---
    Scalar value0 = qsa.value(state0, action0);
    Scalar expected = discreteModel.actions(state1).flatten(0) //
        .map(action1 -> policyInterface.policy(state1, action1).multiply(qsa.value(state1, action1))) //
        .reduce(Scalar::add).get();
    qsa.increment(state0, action0, //
        reward.add(expected.multiply(gamma)).subtract(value0).multiply(alpha));
  }
}
