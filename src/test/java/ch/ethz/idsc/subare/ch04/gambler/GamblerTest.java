// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class GamblerTest extends TestCase {
  public void testActions() {
    Gambler gambler = new Gambler(100, RealScalar.of(0.4));
    assertEquals(gambler.actions(RealScalar.ZERO), Tensors.vector(0));
    assertEquals(gambler.actions(RealScalar.of(1)), Tensors.vector(1));
    assertEquals(gambler.actions(RealScalar.of(2)), Tensors.vector(1, 2));
    assertEquals(gambler.actions(RealScalar.of(100)), Tensors.vector(0));
  }

  public void testActions2() {
    assertEquals(new Gambler(10, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2, 3));
    assertEquals(new Gambler(5, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2));
  }
}
