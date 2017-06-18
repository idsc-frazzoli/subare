// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateRaster;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Dimensions;

public class DynamazeRaster implements StateRaster {
  private final Dynamaze dynamaze;

  public DynamazeRaster(Dynamaze dynamaze) {
    this.dynamaze = dynamaze;
  }

  @Override
  public DiscreteModel discreteModel() {
    return dynamaze;
  }

  @Override
  public Dimension dimension() {
    List<Integer> dimensions = Dimensions.of(dynamaze.image());
    return new Dimension(dimensions.get(0), dimensions.get(1));
  }

  @Override
  public Point point(Tensor state) {
    return new Point( //
        state.Get(0).number().intValue(), //
        state.Get(1).number().intValue());
  }
}
