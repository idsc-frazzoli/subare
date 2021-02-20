// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.util.AssertFail;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.mat.Tolerance;
import junit.framework.TestCase;

public class LinearExplorationRateTest extends TestCase {
  public void testSimple() {
    LinearExplorationRate.of(10, 1, .5);
    LinearExplorationRate.of(10, 1, 1);
    AssertFail.of(() -> LinearExplorationRate.of(10, .1, .2));
  }

  public void testValue() {
    LinearExplorationRate explorationRate = (LinearExplorationRate) LinearExplorationRate.of(10, 0.7, .2);
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(0)), RealScalar.of(0.7));
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(5)), RealScalar.of(0.45));
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(10)), RealScalar.of(0.2));
  }
}
