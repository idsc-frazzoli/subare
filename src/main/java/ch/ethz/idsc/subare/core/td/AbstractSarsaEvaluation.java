// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class AbstractSarsaEvaluation implements SarsaEvaluation {
  final DiscreteModel discreteModel;

  public AbstractSarsaEvaluation(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override
  public final Scalar evaluate(Tensor state, PolicyBase policy) {
    return discreteModel.actions(state).stream() //
        .anyMatch(action -> policy.sac().isEncountered(StateAction.key(state, action))) //
            ? crossEvaluate(state, policy, policy)
            : RealScalar.ZERO;
  }

  @Override
  public Scalar crossEvaluate(Tensor state, PolicyBase policy1, PolicyBase policy2) {
    Tensor action = new PolicyWrap(policy1).next(state, discreteModel);
    return policy2.qsaInterface().value(state, action);
  }
}
