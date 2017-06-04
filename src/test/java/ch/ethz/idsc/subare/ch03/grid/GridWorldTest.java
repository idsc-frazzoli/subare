// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class GridWorldTest extends TestCase {
  public void testBasics() {
    Gridworld gw = new Gridworld();
    assertEquals(gw.reward(Tensors.vector(0, 0), Tensors.vector(1, 0), null), RealScalar.ZERO);
    assertEquals(gw.reward(Tensors.vector(0, 0), Tensors.vector(-1, 0), null), RealScalar.ONE.negate());
  }

  public void testIndex() {
    Gridworld gw = new Gridworld();
    Index actionsIndex = Index.build(gw.actions);
    int index = actionsIndex.of(Tensors.vector(1, 0));
    assertEquals(index, 3);
  }
}
