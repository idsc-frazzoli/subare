// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Expected Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.9) on p.142 */
public class ExpectedSarsa extends Sarsa {
  /** @param discreteModel
   * @param qsa
   * @param learningRate */
  public ExpectedSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override
  protected Scalar evaluate(Tensor state) {
    return crossEvaluate(state, qsa);
  }

  @Override
  protected Scalar crossEvaluate(Tensor state, QsaInterface Qsa2) {
    return discreteModel.actions(state).flatten(0) //
        .map(action -> policy.probability(state, action).multiply(Qsa2.value(state, action))) //
        .reduce(Scalar::add).get();
  }
}
