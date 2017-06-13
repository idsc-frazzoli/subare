// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Expected Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.9) on p.142 */
public class ExpectedSarsa extends Sarsa {
  /** @param discreteModel
   * @param qsa
   * @param alpha if all state transtions are deterministic and all randomness comes
   * from the policy then alpha can be set to 1
   * @param policyInterface */
  public ExpectedSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override
  protected Scalar evaluate(Tensor state) {
    return discreteModel.actions(state).flatten(0) //
        .map(action1 -> policyInterface.policy(state, action1).multiply(qsa.value(state, action1))) //
        .reduce(Scalar::add).get();
  }
}
