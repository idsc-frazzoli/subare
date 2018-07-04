// code by jph
package ch.ethz.idsc.subare.demo.virtualstations;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class StaticHelperTest extends TestCase {
  public void testSimple() {
    Tensor prefix = Tensors.empty();
    Tensor result = StaticHelper.binaryVectors(3, prefix);
    assertEquals(result.length(), 8);
  }

  public void testBaseCase() {
    Tensor prefix = Tensors.empty();
    Tensor result = StaticHelper.binaryVectors(0, prefix);
    assertEquals(result.length(), 0);
  }

  public void testZeroVectors() {
    Tensor prefix = Tensors.empty();
    Tensor result = StaticHelper.zeroVectors(8, prefix);
    // System.out.println(result);
    assertEquals(result.length(), 2);
  }
}
