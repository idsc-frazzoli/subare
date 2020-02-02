// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.ch04.gambler.GamblerModel;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import junit.framework.TestCase;

public class ExactFeatureMapperTest extends TestCase {
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = GamblerModel.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }

  public void testFail() {
    try {
      ExactFeatureMapper.of(null);
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
