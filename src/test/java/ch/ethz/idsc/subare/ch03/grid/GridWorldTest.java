// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import junit.framework.TestCase;

public class GridWorldTest extends TestCase {
  public void testBasics() {
    GridWorld gw = new GridWorld();
    assertEquals(gw.reward(Tensors.vector(0, 0), Tensors.vector(1, 0)), ZeroScalar.get());
    assertEquals(gw.reward(Tensors.vector(0, 0), Tensors.vector(-1, 0)), RealScalar.ONE.negate());
  }

  public void testIndex() {
    GridWorld gw = new GridWorld();
    Index actionsIndex = Index.build(gw.actions);
    int index = actionsIndex.of(Tensors.vector(1, 0));
    assertEquals(index, 3);
  }
}
