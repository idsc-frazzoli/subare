// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import junit.framework.TestCase;

public class PoissonDistributionTest extends TestCase {
  public static void main(String[] args) {
    PoissonDistribution poissonDistribution = PoissonDistribution.of(RealScalar.of(2));
    // Scalar expected = poissonDistribution.values.dot(Range.of(0, poissonDistribution.values.length())).Get();
    // System.out.println(expected);
  }
}
