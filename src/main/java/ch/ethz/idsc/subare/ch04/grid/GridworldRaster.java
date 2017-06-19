// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateActionRaster;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class GridworldRaster implements StateActionRaster {
  private final Gridworld gridworld;
  private final Index indexActions;

  public GridworldRaster(Gridworld gridworld) {
    this.gridworld = gridworld;
    indexActions = Index.build(gridworld.actions);
  }

  @Override
  public DiscreteModel discreteModel() {
    return gridworld;
  }

  @Override
  public Dimension dimension() {
    return new Dimension((gridworld.NX + 1) * 4 - 1, gridworld.NY);
  }

  @Override
  public Point point(Tensor state, Tensor action) {
    int sx = state.Get(0).number().intValue();
    int sy = state.Get(1).number().intValue();
    int a = indexActions.of(action);
    return new Point(sx + (gridworld.NX + 1) * a, sy);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(1);
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.of(15);
  }

  @Override
  public int joinAlongDimension() {
    return 1;
  }

  @Override
  public int magify() {
    return 6;
  }
}
