// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.util.AssertFail;
import junit.framework.TestCase;

public class LinearExplorationRateTest extends TestCase {
  public void testSimple() {
    LinearExplorationRate.of(10, 1, .5);
    LinearExplorationRate.of(10, 1, 1);
    AssertFail.of(() -> LinearExplorationRate.of(10, .1, .2));
  }
}
