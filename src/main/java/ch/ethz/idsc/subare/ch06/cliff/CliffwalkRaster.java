// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateActionRaster;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class CliffwalkRaster implements StateActionRaster {
  private final Cliffwalk cliffwalk;
  private final Index indexActions;

  public CliffwalkRaster(Cliffwalk cliffwalk) {
    this.cliffwalk = cliffwalk;
    indexActions = Index.build(Cliffwalk.ACTIONS);
  }

  @Override
  public DiscreteModel discreteModel() {
    return cliffwalk;
  }

  @Override
  public Dimension dimension() {
    return new Dimension(cliffwalk.NX, (cliffwalk.NY + 1) * 4 - 1);
  }

  @Override
  public Point point(Tensor state, Tensor action) {
    int sx = state.Get(0).number().intValue();
    int sy = state.Get(1).number().intValue();
    int a = indexActions.of(action);
    return new Point(sx, sy + (cliffwalk.NY + 1) * a);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(100);
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.of(15);
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magify() {
    return 3;
  }
}
