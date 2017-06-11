// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.Tensor;

public interface StateActionRaster {
  /** @return underlying discrete model */
  DiscreteModel discreteModel();

  /** @return dimension of raster */
  Dimension dimension();

  /** @param state
   * @param action
   * @return point with x, y as coordinates of state-action pair in raster,
   * or null if state-action pair does not have a position in the raster */
  Point point(Tensor state, Tensor action);
}
