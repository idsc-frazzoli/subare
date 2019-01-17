// code by jph
package ch.ethz.idsc.subare.demo.bus;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import junit.framework.TestCase;

public class SawtoothTest extends TestCase {
  public void testSimple() {
    Sawtooth sawtooth = new Sawtooth(3);
    Tensor s = Range.of(0, 12).map(sawtooth);
    assertEquals(s, Tensors.vector(0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1));
  }
}
