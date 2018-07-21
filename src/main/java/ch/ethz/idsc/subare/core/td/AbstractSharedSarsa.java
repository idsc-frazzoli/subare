// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

abstract class AbstractSharedSarsa extends Sarsa {
  AbstractSharedSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override // from Sarsa
  final Scalar evaluate(Tensor state) {
    return crossEvaluate(state, qsa);
  }

  @Override // from Sarsa
  final Scalar crossEvaluate(Tensor state, Tensor actions, QsaInterface Qsa2) {
    Policy policy = EGreedyPolicy.bestEquiprobable(discreteModel, Qsa2, epsilon, state);
    return crossEvaluate(state, actions, Qsa2, policy);
  }

  abstract Scalar crossEvaluate(Tensor state, Tensor actions, QsaInterface Qsa2, Policy policy);
}
