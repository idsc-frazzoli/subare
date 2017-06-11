// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateRaster;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Dimensions;

class WireloopRaster implements StateRaster {
  private final Wireloop wireloop;

  public WireloopRaster(Wireloop wireloop) {
    this.wireloop = wireloop;
  }

  @Override
  public DiscreteModel discreteModel() {
    return wireloop;
  }

  @Override
  public Dimension dimension() {
    List<Integer> list = Dimensions.of(wireloop.image());
    return new Dimension(list.get(0), list.get(1));
  }

  @Override
  public Point point(Tensor state) {
    return new Point( //
        state.Get(0).number().intValue(), //
        state.Get(1).number().intValue());
  }
}
