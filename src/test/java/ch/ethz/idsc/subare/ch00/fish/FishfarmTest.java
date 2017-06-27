// code by jph
package ch.ethz.idsc.subare.ch00.fish;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Range;
import junit.framework.TestCase;

public class FishfarmTest extends TestCase {
  public void testSimple() {
    Fishfarm fishfarm = new Fishfarm(20, 10);
    // System.out.println(Pretty.of(fishfarm.states()));
  }

  public void testActions() {
    Fishfarm fishfarm = new Fishfarm(20, 10);
    Tensor actions = fishfarm.actions(Tensors.vector(2, 6));
    // System.out.println(actions);
    assertEquals(actions, Range.of(0, 6 + 1));
  }

  public void testMove() {
    Fishfarm fishfarm = new Fishfarm(20, 10);
    Tensor next = fishfarm.move(Tensors.vector(2, 10), RealScalar.of(1));
    // System.out.println(next);
  }

  public void testGrowth() {
    int n = 20;
    Fishfarm fishfarm = new Fishfarm(1, n);
    Tensor res = Range.of(0, n + 1).map(fishfarm::growth);
    // System.out.println(res);
  }
}
