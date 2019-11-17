// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ch.ethz.idsc.subare.core.MoveInterface;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;

/** for deterministic {@link MoveInterface} to precompute and store the effective actions */
public class StateActionMap {
  private final Map<Tensor, Tensor> map;

  public StateActionMap(Map<Tensor, Tensor> map) {
    this.map = map;
  }

  public StateActionMap() {
    this(new HashMap<>());
  }

  /** @param state
   * @return unmodifiable tensor of actions */
  public Tensor actions(Tensor state) {
    return Objects.requireNonNull(map.get(state));
  }

  /** @param state
   * @param actions
   * @throws Exception if state already exists as key in this map */
  public void put(Tensor state, Tensor actions) {
    if (map.containsKey(state))
      throw TensorRuntimeException.of(state);
    map.put(state, actions.unmodifiable());
  }
}
