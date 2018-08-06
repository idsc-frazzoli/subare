// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/* package */ class ExpectedSarsaEvaluation extends AbstractSarsaEvaluation {
  public ExpectedSarsaEvaluation(DiscreteModel discreteModel) {
    super(discreteModel);
  }

  @Override
  public Scalar crossEvaluate(Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2) {
    Policy policy = EGreedyPolicy.bestEquiprobable(discreteModel, qsa1, epsilon, state);
    return actions.stream() //
        .map(action -> policy.probability(state, action).multiply(qsa2.value(state, action))) //
        .reduce(Scalar::add).get();
  }
}
