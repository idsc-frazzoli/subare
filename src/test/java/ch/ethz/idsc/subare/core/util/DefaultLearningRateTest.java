// code by jph
package ch.ethz.idsc.subare.core.util;

import junit.framework.TestCase;

public class DefaultLearningRateTest extends TestCase {
  public void testFailFactor() {
    try {
      DefaultLearningRate.of(0, 1);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
    try {
      DefaultLearningRate.of(-1, 1);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }

  public void testFailExponent() {
    try {
      DefaultLearningRate.of(1, 0.5);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
    try {
      DefaultLearningRate.of(1, 0.4);
      assertTrue(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
