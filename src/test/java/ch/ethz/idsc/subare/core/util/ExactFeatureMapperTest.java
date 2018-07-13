// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.ch04.gambler.Gambler;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import junit.framework.TestCase;

public class ExactFeatureMapperTest extends TestCase {
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = Gambler.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }

  public void testFail() {
    try {
      ExactFeatureMapper.of(null);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
