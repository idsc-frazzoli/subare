// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.Tensor;

public abstract class DefaultStateRaster implements StateRaster {
  protected final DiscreteModel discreteModel;
  private final Dimension dimension;

  public DefaultStateRaster(DiscreteModel discreteModel, List<Integer> list) {
    this.discreteModel = discreteModel;
    dimension = new Dimension(list.get(0), list.get(1));
  }

  @Override
  public final DiscreteModel discreteModel() {
    return discreteModel;
  }

  @Override
  public final Dimension dimension() {
    return dimension;
  }

  @Override
  public Point point(Tensor state) { // TODO remove final after testing
    return new Point( //
        state.Get(0).number().intValue(), //
        state.Get(1).number().intValue());
  }
}