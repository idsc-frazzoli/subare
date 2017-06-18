// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.Tensor;

class StepSet {
  private final Map<Tensor, StepInterface> map = new HashMap<>();

  void register(StepInterface stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    if (!map.containsKey(key))
      map.put(key, stepInterface);
  }

  Collection<StepInterface> values() {
    return map.values();
  }
}
