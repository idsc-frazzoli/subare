// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import junit.framework.TestCase;

public class CliffwalkTest extends TestCase {
  public void testStates() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertEquals(cliffwalk.states().length(), 12 * 3 + 2);
  }

  public void testMove() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertEquals(cliffwalk.move(Tensors.vector(0, 0), Tensors.vector(0, -1)), Tensors.vector(0, 0));
    assertEquals(cliffwalk.move(Tensors.vector(0, 0), Tensors.vector(0, 1)), Tensors.vector(0, 1));
    assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, 1)), Tensors.vector(0, 3));
    assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, -1)), Tensors.vector(0, 2));
    assertEquals(cliffwalk.move(Tensors.vector(11, 3), Tensors.vector(0, -1)), Tensors.vector(11, 3));
  }

  public void testCliff() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertFalse(cliffwalk.isCliff(Tensors.vector(0, 3)));
    assertTrue(cliffwalk.isCliff(Tensors.vector(1, 3)));
    assertTrue(cliffwalk.isCliff(Tensors.vector(10, 3)));
    assertFalse(cliffwalk.isCliff(Tensors.vector(11, 3)));
  }

  public void testReward() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertEquals(cliffwalk.reward(Tensors.vector(11, 2), null, Tensors.vector(11, 3)), RealScalar.ONE);
    assertEquals(cliffwalk.reward(Tensors.vector(11, 3), null, Tensors.vector(11, 3)), ZeroScalar.get());
    // assertEquals(cliffwalk.move(, Tensors.vector(0, -1)), Tensors.vector(0, 0));
    // assertEquals(cliffwalk.move(Tensors.vector(0, 0), Tensors.vector(0, 1)), Tensors.vector(0, 1));
    // assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, 1)), Tensors.vector(0, 3));
    // assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, -1)), Tensors.vector(0, 2));
    // assertEquals(cliffwalk.move(Tensors.vector(11, 3), Tensors.vector(0, -1)), Tensors.vector(11, 3));
  }
}
