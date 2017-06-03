// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import junit.framework.TestCase;

public class CarRentalTest extends TestCase {
  public void testActions() {
    CarRental cr = new CarRental(true);
    assertEquals(cr.actions(Tensors.vector(3, 1)), Range.of(-3, 1 + 1));
    // System.out.println(Pretty.of(cr.actions(Tensors.vector(3, 1))));
  }

  public void testActions2() {
    CarRental cr = new CarRental(true);
    assertEquals(cr.actions(Tensors.vector(10, 10)), Range.of(-5, 5 + 1));
    // System.out.println(Pretty.of(cr.actions(Tensors.vector(10, 10))));
  }
}
