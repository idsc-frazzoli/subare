// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** measure to compare performance between optimal state-action value function and learned q-function */
public enum Loss {
  ;
  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return non-negative number which should be subtracted from the optimal gains */
  public static Scalar accumulation(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    Scalar sum = RealScalar.ZERO;
    for (Tensor state : discreteModel.states()) {
      Tensor actions = discreteModel.actions(state);
      Scalar max = actions.flatten(0) //
          .map(action -> ref.value(state, action)) //
          .reduce(Max::of).get();
      FairArgMax fairArgMax = FairArgMax.of(Tensor.of(actions.flatten(0).map(action -> qsa.value(state, action))));
      Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount());
      for (int index : fairArgMax.options()) {
        Scalar delta = max.subtract(ref.value(state, actions.get(index)));
        sum = sum.add(delta.multiply(weight));
      }
    }
    return sum;
  }
}
