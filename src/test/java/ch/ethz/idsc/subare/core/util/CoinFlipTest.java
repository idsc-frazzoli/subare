package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.util.CoinFlip;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Chop;
import junit.framework.TestCase;

public class CoinFlipTest extends TestCase {
  public void testProbabilityDistribution() {
    Scalar headProbability0 = RealScalar.of(0.1);
    Scalar headProbability1 = RealScalar.of(0.5);
    Scalar headProbability2 = RealScalar.of(0.9);
    CoinFlip coinFlip0 = CoinFlip.of(headProbability0);
    CoinFlip coinFlip1 = CoinFlip.of(headProbability1);
    CoinFlip coinFlip2 = CoinFlip.of(headProbability2);
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
      CoinFlip.of(RealScalar.of(-0.1));
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
