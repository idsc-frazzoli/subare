// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class StateActionMapTest extends TestCase {
  public void testSimple() {
    StateActionMap stateActionMap = new StateActionMap();
    Tensor key = Tensors.vector(1);
    Tensor values = Tensors.vector(1, 2);
    stateActionMap.put(key, values);
    Tensor actions = stateActionMap.actions(Tensors.vector(1));
    assertEquals(actions, values);
  }

  public void testDuplicateFail() {
    StateActionMap stateActionMap = new StateActionMap();
    stateActionMap.put(Tensors.vector(1), Tensors.vector(1, 2));
    try {
      stateActionMap.put(Tensors.vector(1), Tensors.vector(1, 2));
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
