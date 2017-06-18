// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.BinaryOperator;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;

public enum DiscreteUtils {
  ;
  /** @param discreteModel
   * @return index for state-action */
  public static Index build(DiscreteModel discreteModel, Tensor states) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : discreteModel.actions(state))
        tensor.append(Tensors.of(state, action));
    return Index.build(tensor);
  }

  // ---
  /** @param discreteModel
   * @param qsa
   * @param binaryOperator
   * @return */
  public static DiscreteVs reduce( //
      DiscreteModel discreteModel, QsaInterface qsa, BinaryOperator<Scalar> binaryOperator) {
    return DiscreteVs.build(discreteModel, //
        Tensor.of(discreteModel.states().flatten(0) //
            .map(state -> discreteModel.actions(state).flatten(0) //
                .map(action -> qsa.value(state, action)) //
                .reduce(binaryOperator).get()))); // <- assumes greedy policy
  }

  /** compute state value function v(s) based on given action-value function q(s,a)
   * 
   * @param discreteModel
   * @param qsa
   * @return state values */
  public static DiscreteVs createVs(DiscreteModel discreteModel, QsaInterface qsa) {
    return reduce(discreteModel, qsa, Max::of);
  }
}
