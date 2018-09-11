// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

class AbstractSarsaEvaluation implements SarsaEvaluation {
  final DiscreteModel discreteModel;

  public AbstractSarsaEvaluation(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override
  public final Scalar evaluate(Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa, StateActionCounter sac) {
    Tensor actions = Tensor.of( //
        discreteModel.actions(state).stream() //
            .filter(action -> sac.isEncountered(StateAction.key(state, action))));
    return Tensors.isEmpty(actions) //
        ? RealScalar.ZERO
        : crossEvaluate(epsilon, state, actions, qsa, qsa);
  }

  @Override
  public Scalar crossEvaluate(Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2) {
    Policy policy = new EGreedyPolicy(discreteModel, qsa1, epsilon, state);
    Tensor action = new PolicyWrap(policy).next(state, actions);
    return qsa2.value(state, action);
  }
}
