// code by fluric
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class CoinflipTest extends TestCase {
  public void testProbabilityDistribution() {
    Scalar headProbability0 = RealScalar.of(0.1);
    Scalar headProbability1 = RealScalar.of(0.5);
    Scalar headProbability2 = RealScalar.of(0.9);
    Coinflip coinFlip0 = Coinflip.of(headProbability0);
    Coinflip coinFlip1 = Coinflip.of(headProbability1);
    Coinflip coinFlip2 = Coinflip.of(headProbability2);
    int[] counters = { 0, 0, 0 };
    int rounds = 100000;
    for (int i = 0; i < rounds; ++i) {
      counters[0] += coinFlip0.tossHead() ? 1 : 0;
      counters[1] += coinFlip1.tossHead() ? 1 : 0;
      counters[2] += coinFlip2.tossHead() ? 1 : 0;
    }
    assertTrue(Chop.below(1e-2).close(RationalScalar.of(counters[0], rounds), headProbability0));
    assertTrue(Chop.below(1e-2).close(RationalScalar.of(counters[1], rounds), headProbability1));
    assertTrue(Chop.below(1e-2).close(RationalScalar.of(counters[2], rounds), headProbability2));
  }

  public void testFail() {
    try {
      Coinflip.of(RealScalar.of(-0.1));
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
    try {
      Coinflip.of(RealScalar.of(1.1));
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
