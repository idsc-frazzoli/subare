// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import junit.framework.TestCase;

public class RandomwalkTest extends TestCase {
  public void testSmall() {
    Randomwalk rw = new Randomwalk(5);
    assertEquals(rw.states(), Range.of(0, 7));
    assertEquals(rw.startStates(), Tensors.vector(3));
    assertTrue(rw.isTerminal(RealScalar.of(6)));
  }

  public void testLarge() {
    Randomwalk rw = new Randomwalk(19);
    assertEquals(rw.states(), Range.of(0, 21));
    assertEquals(rw.startStates(), Tensors.vector(10));
    assertTrue(rw.isTerminal(RealScalar.of(20)));
  }
}
