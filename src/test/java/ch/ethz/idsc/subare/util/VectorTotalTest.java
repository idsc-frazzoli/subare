// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.mat.HilbertMatrix;
import junit.framework.TestCase;

public class VectorTotalTest extends TestCase {
  public void testSimple() {
    Scalar scalar = VectorTotal.FUNCTION.apply(Tensors.vector(1, 2, 3));
    assertEquals(scalar, RealScalar.of(6));
  }

  public void testFail() {
    try {
      VectorTotal.FUNCTION.apply(HilbertMatrix.of(3));
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
