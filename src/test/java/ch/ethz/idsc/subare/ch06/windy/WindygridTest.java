// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class WindygridTest extends TestCase {
  @SuppressWarnings("unused")
  public void testWindy() {
    Windygrid windyGrid = Windygrid.createFour();
    Tensor state = Tensors.vector(6, 0);
    for (Tensor action : windyGrid.actions(state)) {
      // System.out.println(action + " -> " + windyGrid.move(state, action));
    }
  }

  @SuppressWarnings("unused")
  public void testRepmat() {
    // System.out.println("one right from goal");
    Windygrid windyGrid = Windygrid.createFour();
    Tensor right = Tensors.vector(1, 0);
    Tensor left = Tensors.vector(-1, 0);
    Tensor up = Tensors.vector(0, 1);
    Tensor state = Windygrid.GOAL.add(up);
    // System.out.println(state);
    Tensor dest = windyGrid.move(state, left);
    assertEquals(dest, Tensors.vector(6, 2));
    // System.out.println("left = " + dest);
    for (Tensor action : windyGrid.actions(state)) {
      // System.out.println();
      // System.out.println(action + " -> " + windyGrid.move(state, action));
    }
  }
}
