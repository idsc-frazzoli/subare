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
 * box on p.140
 * 
 * see also Watkins 1989 */
public class QLearning extends Sarsa {
  /** @param discreteModel
   * @param qsa
   * @param alpha learning rate should converge to zero, with
   * sum of alpha's should go to infinity, sum of alpha's squared should be finite */
  public QLearning(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override
  protected Scalar evaluate(Tensor state) {
    return discreteModel.actions(state).flatten(0) //
        .map(action -> qsa.value(state, action)) //
        .reduce(Max::of).get();
  }

  @Override
  protected Scalar crossEvaluate(Tensor state, QsaInterface Qsa2) {
    Scalar value = RealScalar.ZERO;
    Tensor actions = discreteModel.actions(state);
    // use qsa == Qsa1 to determine best actions
    FairArgMax fairArgMax = FairArgMax.of(Tensor.of(actions.flatten(0).map(action -> qsa.value(state, action))));
    Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount()); // uniform distribution among best actions
    for (int index : fairArgMax.options()) {
      Tensor action = actions.get(index);
      value = value.add(Qsa2.value(state, action).multiply(weight)); // use Qsa2 to evaluate state-action pair
    }
    return value;
  }
}
