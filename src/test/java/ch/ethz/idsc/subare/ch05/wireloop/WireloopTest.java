// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.awt.Dimension;
import java.util.Arrays;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.red.Entrywise;
import ch.ethz.idsc.tensor.red.Max;
import junit.framework.TestCase;

public class WireloopTest extends TestCase {
  public void testSimple() throws Exception {
    String name = "wirec";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    assertEquals(Dimensions.of(wireloop.states()), Arrays.asList(297, 2));
    assertEquals(wireloop.states().stream().reduce(Entrywise.max()).get(), Tensors.vector(15, 23));
    assertEquals(RealScalar.of(15), //
        wireloop.states().get(Tensor.ALL, 0).stream().reduce(Max::of).get());
    assertEquals(RealScalar.of(23), //
        wireloop.states().get(Tensor.ALL, 1).stream().reduce(Max::of).get());
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Dimension dimension = wireloopRaster.dimensionStateRaster();
    assertEquals(dimension.width, 16);
    assertEquals(dimension.height, 24);
  }
}
