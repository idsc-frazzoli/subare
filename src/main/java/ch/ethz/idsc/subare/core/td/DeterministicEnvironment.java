// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** utility class to implement "Model" for deterministic environments
 * in Tabular Dyna-Q p.172 */
/* package */ class DeterministicEnvironment implements StepDigest {
  private static final Random RANDOM = new Random();
  // ---
  private final Map<Tensor, StepInterface> map = new HashMap<>();
  private final Tensor keys = Tensors.empty();

  public StepInterface getRandomStep() {
    return map.get(keys.get(RANDOM.nextInt(size())));
  }

  public StepInterface get(Tensor state, Tensor action) {
    return map.get(StateAction.key(state, action));
  }

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    register(key, stepInterface);
  }

  private synchronized void register(Tensor key, StepInterface stepInterface) {
    if (!map.containsKey(key)) {
      map.put(key, stepInterface);
      keys.append(key); // after updating the map, for conservative size
    } else {
      // TODO can verify that stored step is identical to provided step
    }
  }

  public int size() {
    return keys.length();
  }
}
