// code by jph
package ch.ethz.idsc.subare.ch00.bus;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class ChargerTest extends TestCase {
  public void testSimple() {
    Charger charger = new Charger(5, 3);
    assertTrue(charger.isTerminal(Tensors.vector(4, 0)));
    Tensor actions = charger.actions(Tensors.vector(0, 0));
    assertEquals(actions.length(), 4);
    Tensor res = charger.move(Tensors.vector(2, 2), Tensors.vector(1, 3));
    assertEquals(res, Tensors.vector(3, 2));
  }
}
