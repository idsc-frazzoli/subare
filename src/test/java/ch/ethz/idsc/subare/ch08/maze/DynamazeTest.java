// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import junit.framework.TestCase;

public class DynamazeTest extends TestCase {
  public void testSimple() throws Exception {
    Dynamaze dynamaze = DynamazeHelper.create("maze2");
    assertEquals(dynamaze.startStates().length(), 1);
  }
}
