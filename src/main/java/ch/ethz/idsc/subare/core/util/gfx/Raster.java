// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import java.io.Serializable;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.Scalar;

interface Raster extends Serializable {
  /** @return underlying discrete model */
  DiscreteModel discreteModel();

  /** @return loss function scale for visualization */
  Scalar scaleLoss();

  /** @return q function error scale for visualization */
  Scalar scaleQdelta();

  /** @return dimension to join q function, loss, etc. */
  int joinAlongDimension();

  /** @return magnification */
  int magnify();
}
