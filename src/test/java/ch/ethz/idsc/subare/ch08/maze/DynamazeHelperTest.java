// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import java.util.Arrays;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Dimensions;
import junit.framework.TestCase;

public class DynamazeHelperTest extends TestCase {
  public void testMaze2() throws Exception {
    Tensor image = DynamazeHelper.load("maze2");
    assertEquals(Dimensions.of(image), Arrays.asList(6, 9, 4));
  }

  public void testMaze5() throws Exception {
    Tensor image = DynamazeHelper.load("maze5");
    assertEquals(Dimensions.of(image), Arrays.asList(32, 16, 4));
  }
}
