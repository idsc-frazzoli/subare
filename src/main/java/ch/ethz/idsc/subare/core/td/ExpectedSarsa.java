// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Expected Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.9) on p.133 */
/* package */ class ExpectedSarsa extends AbstractSharedSarsa {
  ExpectedSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override // from AbstractSharedSarsa
  Scalar crossEvaluate(Tensor state, Tensor actions, QsaInterface Qsa2, Policy policy) {
    return actions.stream() //
        .map(action -> policy.probability(state, action).multiply(Qsa2.value(state, action))) //
        .reduce(Scalar::add).get();
  }
}
