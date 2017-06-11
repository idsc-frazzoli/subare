// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class MaxbiasTest extends TestCase {
  public void testMove() {
    Maxbias maxbias = new Maxbias(3);
    assertEquals(maxbias.move(RealScalar.ONE, RealScalar.ONE), RealScalar.ZERO);
    assertEquals(maxbias.move(RealScalar.of(2), RealScalar.of(1)), RealScalar.of(3));
    assertEquals(maxbias.move(RealScalar.of(2), RealScalar.of(-1)), RealScalar.of(1));
  }

  public void testTerminal() {
    Maxbias maxbias = new Maxbias(3);
    assertTrue(maxbias.isTerminal(RealScalar.ZERO));
    assertFalse(maxbias.isTerminal(RealScalar.ONE));
    assertFalse(maxbias.isTerminal(RealScalar.of(2)));
    assertTrue(maxbias.isTerminal(RealScalar.of(3)));
  }

  public void testReward() {
    Maxbias maxbias = new Maxbias(3);
    assertEquals(maxbias.reward(RealScalar.of(2), RealScalar.of(1), RealScalar.of(3)), RealScalar.ZERO);
  }

  public void testStarting() {
    Maxbias maxbias = new Maxbias(3);
    assertEquals(maxbias.startStates(), Tensors.vector(2));
  }
}
