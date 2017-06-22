// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import java.awt.Point;

import ch.ethz.idsc.tensor.Tensor;

public interface StateActionRaster extends Raster {
  /** @param state
   * @param action
   * @return point with x, y as coordinates of state-action pair in raster,
   * or null if state-action pair does not have a position in the raster */
  Point point(Tensor state, Tensor action);
}
