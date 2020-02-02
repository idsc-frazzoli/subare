// code by jph, fluric
package ch.ethz.idsc.subare.core.td;

import java.util.Objects;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.PolicyExt;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.subare.util.FairArg;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;

/* package */ class QLearningSarsaEvaluation implements SarsaEvaluation {
  private final DiscreteModel discreteModel;

  public QLearningSarsaEvaluation(DiscreteModel discreteModel) {
    this.discreteModel = Objects.requireNonNull(discreteModel);
  }

  @Override // from SarsaEvaluation
  public Scalar evaluate(Tensor state, PolicyExt policy) {
    return discreteModel.actions(state).stream() //
        .filter(action -> policy.sac().isEncountered(StateAction.key(state, action))) //
        .map(action -> policy.qsaInterface().value(state, action)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
  }

  @Override // from SarsaEvaluation
  public Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2) {
    Scalar value = RealScalar.ZERO;
    Tensor actions = Tensor.of(discreteModel.actions(state).stream() //
        .filter(action -> policy1.sac().isEncountered(StateAction.key(state, action))));
    if (Tensors.isEmpty(actions))
      return RealScalar.ZERO;
    Tensor eval = Tensor.of(actions.stream().map(action -> policy1.qsaInterface().value(state, action)));
    FairArg fairArgMax = FairArg.max(eval);
    Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount()); // uniform distribution among best actions
    for (int index : fairArgMax.options()) {
      Tensor action = actions.get(index);
      value = value.add(policy2.qsaInterface().value(state, action).multiply(weight)); // use Qsa2 to evaluate state-action pair
    }
    return value;
  }
}
