// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.awt.Dimension;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;
import junit.framework.TestCase;

public class WireloopTest extends TestCase {
  public void testSimple() throws Exception {
    String name = "wirec";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    assertEquals(RealScalar.of(15), //
        wireloop.states().get(Tensor.ALL, 0).flatten(0).reduce(Max::of).get().Get());
    assertEquals(RealScalar.of(23), //
        wireloop.states().get(Tensor.ALL, 1).flatten(0).reduce(Max::of).get().Get());
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Dimension dimension = wireloopRaster.dimensionStateRaster();
    assertEquals(dimension.width, 16);
    assertEquals(dimension.height, 24);
  }
}
