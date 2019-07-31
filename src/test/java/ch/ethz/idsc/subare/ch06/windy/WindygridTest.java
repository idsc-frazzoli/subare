// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class WindygridTest extends TestCase {
  public void testWindy() {
    Windygrid windyGrid = Windygrid.createFour();
    Tensor state = Tensors.vector(6, 0);
    windyGrid.actions(state);
  }

  public void testRepmat() {
    Windygrid windyGrid = Windygrid.createFour();
    Tensor left = Tensors.vector(-1, 0);
    Tensor up = Tensors.vector(0, 1);
    Tensor state = Windygrid.GOAL.add(up);
    Tensor dest = windyGrid.move(state, left);
    assertEquals(dest, Tensors.vector(6, 2));
    windyGrid.actions(state);
  }
}
