// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Pretty;
import junit.framework.TestCase;

public class CarRentalTest extends TestCase {
  public void testActions() {
    CarRental cr = new CarRental();
    System.out.println(Pretty.of(cr.actions(Tensors.vector(3, 1))));
  }
}
