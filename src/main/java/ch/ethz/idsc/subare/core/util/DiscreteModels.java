// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public enum DiscreteModels {
  ;
  /** @param discreteModel
   * @return */
  public static Index build(DiscreteModel discreteModel) {
    Tensor qas = Tensors.empty();
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qas.append(Tensors.of(state, action));
    return Index.build(qas);
  }
}
