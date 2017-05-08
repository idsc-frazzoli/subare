// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** for deterministic {@link MoveInterface} to reduce and speed up
 * the computation of the effective actions */
public class StateActionMap {
  /** @param discreteModel
   * @param actions
   * @param moveInterface deterministic
   * @return */
  public static StateActionMap build( //
      DiscreteModel discreteModel, Tensor actions, MoveInterface moveInterface) {
    final Map<Tensor, Tensor> map = new HashMap<>();
    for (Tensor state : discreteModel.states()) {
      Tensor filter = Tensors.empty();
      Set<Tensor> set = new HashSet<>();
      for (Tensor action : actions) {
        Tensor next = moveInterface.move(state, action);
        if (set.add(next))
          filter.append(action);
      }
      if (filter.length() == 0)
        throw new RuntimeException("no actions for " + state);
      map.put(state, filter.unmodifiable());
    }
    return new StateActionMap(map);
  }

  private final Map<Tensor, Tensor> map;

  private StateActionMap(Map<Tensor, Tensor> map) {
    this.map = map;
  }

  public Tensor actions(Tensor state) {
    return map.get(state);
  }
}
