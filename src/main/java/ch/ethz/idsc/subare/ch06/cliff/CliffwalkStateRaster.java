// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateRaster;
import ch.ethz.idsc.tensor.Tensor;

public class CliffwalkStateRaster implements StateRaster {
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
    int sx = state.Get(0).number().intValue();
    int sy = state.Get(1).number().intValue();
    return new Point(sx, sy);
  }
}
