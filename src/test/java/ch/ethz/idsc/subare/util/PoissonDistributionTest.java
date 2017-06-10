// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Total;
import junit.framework.TestCase;

public class PoissonDistributionTest extends TestCase {
  public void testSimple() {
    PoissonDistribution poissonDistribution = PoissonDistribution.of(RealScalar.of(2));
    Tensor prob = poissonDistribution.values(16);
    Scalar scalar = Total.of(prob).Get();
    assertTrue(Scalars.lessThan(RealScalar.of(.9999), scalar));
    assertTrue(Scalars.lessThan(scalar, RealScalar.ONE));
  }

  public void testMemo() {
    int size = PoissonDistribution.MEMO.size();
    {
      PoissonDistribution.of(RealScalar.of(32.));
      PoissonDistribution.of(RealScalar.of(32.));
      PoissonDistribution.of(RealScalar.of(32.));
    }
    assertEquals(size + 1, PoissonDistribution.MEMO.size());
  }

  public void testPrecompute() {
    PoissonDistribution poissonDistribution = PoissonDistribution.of(RealScalar.of(3));
    assertEquals(poissonDistribution.values().length(), PoissonDistribution.PRECOMPUTE_LENGTH);
  }

  public void testValues() {
    PoissonDistribution poissonDistribution = PoissonDistribution.of(RealScalar.of(3));
    poissonDistribution.apply(30);
    assertEquals(poissonDistribution.values().length(), 30 + 1);
    Scalar sum = Total.of(poissonDistribution.values()).Get();
    // System.out.println(sum);
    assertEquals(sum, RealScalar.ONE);
  }
}
