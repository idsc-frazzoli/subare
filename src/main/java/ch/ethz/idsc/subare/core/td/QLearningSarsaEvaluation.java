// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/* package */ class QLearningSarsaEvaluation implements SarsaEvaluation {
  private final DiscreteModel discreteModel;

  public QLearningSarsaEvaluation(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override // from SarsaEvaluation
  public Scalar evaluate(Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa, StateActionCounter stateActionCounter) {
    return discreteModel.actions(state).stream() //
        .filter(action -> stateActionCounter.isEncountered(StateAction.key(state, action))) //
        .map(action -> qsa.value(state, action)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
  }

  @Override // from SarsaEvaluation
  public Scalar crossEvaluate(Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2) {
    Scalar value = RealScalar.ZERO;
    Tensor eval = Tensor.of(actions.stream().map(action -> qsa1.value(state, action)));
    FairArgMax fairArgMax = FairArgMax.of(eval);
    Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount()); // uniform distribution among best actions
    for (int index : fairArgMax.options()) {
      Tensor action = actions.get(index);
      value = value.add(qsa2.value(state, action).multiply(weight)); // use Qsa2 to evaluate state-action pair
    }
    return value;
  }
}
