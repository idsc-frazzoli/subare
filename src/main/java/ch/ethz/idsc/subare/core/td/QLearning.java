// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** Q-learning: An off-policy TD control algorithm
 * 
 * eq (6.8)
 * 
 * box on p.131
 * 
 * see also Watkins 1989 */
/* package */ class QLearning extends Sarsa {
  QLearning(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override // from Sarsa
  Scalar evaluate(Tensor state) {
    return discreteModel.actions(state).stream() //
        .filter(action -> learningRate.encountered(state, action)) //
        .map(action -> qsa.value(state, action)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
  }

  @Override // from Sarsa
  Scalar crossEvaluate(Tensor state, Tensor actions, QsaInterface Qsa2) {
    // TODO untested!!!
    // use qsa == Qsa1 to determine best actions
    Scalar value = RealScalar.ZERO;
    Tensor eval = Tensor.of(actions.stream().map(action -> qsa.value(state, action)));
    FairArgMax fairArgMax = FairArgMax.of(eval);
    Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount()); // uniform distribution among best actions
    for (int index : fairArgMax.options()) {
      Tensor action = actions.get(index);
      value = value.add(Qsa2.value(state, action).multiply(weight)); // use Qsa2 to evaluate state-action pair
    }
    return value;
  }
}
