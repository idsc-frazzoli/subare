// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Expected Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.9) on p.142 */
public class ExpectedSarsa extends Sarsa {
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
      QsaInterface qsa, Scalar alpha) {
    super(episodeSupplier, policyInterface, discreteModel, qsa, alpha);
  }

  @Override
  protected Scalar evaluate(Tensor state1) {
    return discreteModel.actions(state1).flatten(0) //
        .map(action1 -> policyInterface.policy(state1, action1).multiply(qsa.value(state1, action1))) //
        .reduce(Scalar::add).get();
  }
}
