// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import junit.framework.TestCase;

public class TransitionTrackerTest extends TestCase {
  public void testSimple() {
    Map<Integer, Integer> map = new HashMap<>();
    map.merge(10, 101, (i, j) -> i + j);
    map.merge(10, 1, (i, j) -> i + j);
    map.merge(10, 1, (i, j) -> i + j);
    assertEquals(map.get(10).intValue(), 103);
  }
}
