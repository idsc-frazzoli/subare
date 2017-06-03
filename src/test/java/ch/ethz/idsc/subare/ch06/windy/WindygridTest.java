// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class WindygridTest extends TestCase {
  public void testWindy() {
    Windygrid windyGrid = Windygrid.createFour();
    Tensor state = Tensors.vector(6, 0);
    for (Tensor action : windyGrid.actions(state)) {
      // System.out.println(action + " -> " + windyGrid.move(state, action));
    }
  }

  public void testRepmat() {
    // System.out.println("one right from goal");
    Windygrid windyGrid = Windygrid.createFour();
    Tensor right = Tensors.vector(0, 1);
    Tensor state = Windygrid.GOAL.add(right);
    // System.out.println(state);
    Tensor left = Tensors.vector(0, -1);
    Tensor dest = windyGrid.move(state, left);
    assertEquals(dest, Tensors.vector(2, 7));
    // System.out.println("left = " + dest);
    for (Tensor action : windyGrid.actions(state)) {
      // System.out.println();
      // System.out.println(action + " -> " + windyGrid.move(state, action));
    }
  }
}
