// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.tensor.Tensor;

public enum StateActionCounterUtil {
  ;
  /** @param sac1
   * @param sac2
   * @param discreteModel
   * @return new instance of DiscreteStateActionCounter that combines given sac1 and sac2 */
  public static StateActionCounter getSummedSac(StateActionCounter sac1, StateActionCounter sac2, DiscreteModel discreteModel) {
    DiscreteStateActionCounter sac = new DiscreteStateActionCounter();
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Tensor key = StateAction.key(state, action);
        sac.setStateActionCount(key, sac1.stateActionCount(key).add(sac2.stateActionCount(key)));
        sac.setStateCount(state, sac1.stateCount(state).add(sac2.stateCount(state)));
      }
    return sac;
  }
}
