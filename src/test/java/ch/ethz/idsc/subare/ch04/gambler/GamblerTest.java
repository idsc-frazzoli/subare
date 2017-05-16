// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import junit.framework.TestCase;

public class GamblerTest extends TestCase {
  public void testActions() {
    Gambler ga = new Gambler(100, RealScalar.of(0.4));
    assertEquals(ga.actions(ZeroScalar.get()), Tensors.fromString("{0}"));
    assertEquals(ga.actions(RealScalar.of(1)), Tensors.fromString("{1}"));
    assertEquals(ga.actions(ga.TERMINAL_W), Tensors.fromString("{0}"));
  }

  public void testActions2() {
    assertEquals(new Gambler(10, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2, 3));
    assertEquals(new Gambler(5, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2));
  }
}
