// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateRaster;
import ch.ethz.idsc.tensor.Tensor;

class CliffwalkStateRaster implements StateRaster {
  private final Cliffwalk cliffwalk;

  public CliffwalkStateRaster(Cliffwalk cliffwalk) {
    this.cliffwalk = cliffwalk;
  }

  @Override
  public DiscreteModel discreteModel() {
    return cliffwalk;
  }

  @Override
  public Dimension dimension() {
    return new Dimension(cliffwalk.NX, cliffwalk.NY);
  }

  @Override
  public Point point(Tensor state) {
    return new Point( //
        state.Get(0).number().intValue(), //
        state.Get(1).number().intValue());
  }
}
