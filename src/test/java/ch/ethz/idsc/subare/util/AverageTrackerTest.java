// code by jph
package ch.ethz.idsc.subare.util;

import java.util.Objects;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class AverageTrackerTest extends TestCase {
  public void testAverage() {
    AverageTracker avg = new AverageTracker();
    avg.track(RealScalar.of(3));
    assertEquals(avg.getScalar(), RealScalar.of(3));
    avg.track(RealScalar.of(1));
    assertEquals(avg.getScalar(), RealScalar.of(2));
    avg.track(RealScalar.of(1));
    Chop._10.requireClose(avg.getScalar(), RealScalar.of(5. / 3));
  }

  public void testMean() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    AverageTracker avg = new AverageTracker();
    vec.stream().forEach(scalar -> avg.track(scalar));
    assertEquals(avg.getScalar(), Mean.of(vec));
  }

  public void testMean2() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    AverageTracker avg = new AverageTracker();
    vec.stream().map(Scalar.class::cast).forEach(avg::track);
    assertEquals(avg.getScalar(), Mean.of(vec));
  }

  public void testEmpty() {
    AverageTracker avg = new AverageTracker();
    assertTrue(Objects.isNull(avg.get()));
  }
}
