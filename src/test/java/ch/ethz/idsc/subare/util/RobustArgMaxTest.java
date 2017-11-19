// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class RobustArgMaxTest extends TestCase {
  public void testSimple() {
    Tensor tensor = Tensors.vector(-9, 0, 0.9999999, .3, 1, 0.9999999);
    RobustArgMax ram = new RobustArgMax(Chop._06);
    int index = ram.of(tensor);
    assertEquals(index, 2);
  }
}
