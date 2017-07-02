// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class AverageTest extends TestCase {
  public void testAverage() {
    Average avg = new Average();
    avg.track(RealScalar.of(3));
    assertEquals(avg.Get(), RealScalar.of(3));
    avg.track(RealScalar.of(1));
    assertEquals(avg.Get(), RealScalar.of(2));
    avg.track(RealScalar.of(1));
    assertTrue(Chop._10.allZero(avg.Get().subtract(RealScalar.of(5. / 3))));
  }

  public void testMean() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    Average avg = new Average();
    vec.flatten(0).forEach(scalar -> avg.track(scalar));
    assertEquals(avg.Get(), Mean.of(vec));
  }

  public void testEmpty() {
    Average avg = new Average();
    assertTrue(avg.get() == null);
  }
}
