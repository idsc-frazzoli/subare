// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

public enum DiscreteUtils {
  ;
  // ---
  /** @param discreteModel
   * @param qsa
   * @return state values */
  public static DiscreteVs createVs(DiscreteModel discreteModel, QsaInterface qsa) {
    return DiscreteVs.build(discreteModel, //
        Tensor.of(discreteModel.states().flatten(0) //
            .map(state -> discreteModel.actions(state).flatten(0) //
                .map(action -> qsa.value(state, action)) //
                .reduce(Max::of).get()))); // <- assumes greedy policy
  }
}
