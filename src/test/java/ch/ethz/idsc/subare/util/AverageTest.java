package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class AverageTest extends TestCase {
  public void testAverage() {
    Average avg = new Average();
    avg.track(RealScalar.of(3));
    assertEquals(avg.get(), RealScalar.of(3));
    avg.track(RealScalar.of(1));
    assertEquals(avg.get(), RealScalar.of(2));
    avg.track(RealScalar.of(1));
    assertTrue(Scalars.isZero(Chop.function.apply(avg.get().subtract(RealScalar.of(5. / 3)))));
  }
}
