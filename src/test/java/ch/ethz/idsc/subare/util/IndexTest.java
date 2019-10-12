// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.opt.Pi;
import junit.framework.TestCase;

public class IndexTest extends TestCase {
  public void testSimple() {
    Tensor tensor = Tensors.vector(5, 7, 11);
    Index index = Index.build(tensor);
    assertEquals(index.of(RealScalar.of(7)), 1);
    try {
      index.of(RealScalar.of(8));
      fail();
    } catch (Exception exception) {
      // ---
    }
  }

  public void testScalarFail() {
    try {
      Index.build(Pi.VALUE);
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
