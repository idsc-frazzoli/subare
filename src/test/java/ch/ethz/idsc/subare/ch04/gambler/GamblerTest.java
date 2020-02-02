// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class GamblerTest extends TestCase {
  public void testActions() {
    GamblerModel gamblerModel = new GamblerModel(100, RealScalar.of(0.4));
    assertEquals(gamblerModel.actions(RealScalar.ZERO), Tensors.vector(0));
    assertEquals(gamblerModel.actions(RealScalar.of(1)), Tensors.vector(1));
    assertEquals(gamblerModel.actions(RealScalar.of(2)), Tensors.vector(1, 2));
    assertEquals(gamblerModel.actions(RealScalar.of(100)), Tensors.vector(0));
  }

  public void testActions2() {
    assertEquals(new GamblerModel(10, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2, 3));
    assertEquals(new GamblerModel(5, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2));
  }
}
