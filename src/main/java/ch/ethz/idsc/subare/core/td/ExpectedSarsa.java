// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Expected Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.9) on p.133 */
public class ExpectedSarsa extends Sarsa {
  /** @param discreteModel
   * @param qsa
   * @param learningRate */
  public ExpectedSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override // from Sarsa
  protected Scalar evaluate(Tensor state) {
    return crossEvaluate(state, qsa);
  }

  @Override // from Sarsa
  protected Scalar crossEvaluate(Tensor state, QsaInterface Qsa2) {
    Tensor actions = Tensor.of( //
        discreteModel.actions(state).stream() //
            .filter(action -> learningRate.encountered(state, action)));
    if (actions.length() == 0)
      return RealScalar.ZERO;
    // ---
    Policy policy = EGreedyPolicy.bestEquiprobable(discreteModel, Qsa2, epsilon, state);
    return actions.stream() //
        .map(action -> policy.probability(state, action).multiply(Qsa2.value(state, action))) //
        .reduce(Scalar::add).get();
  }
}
