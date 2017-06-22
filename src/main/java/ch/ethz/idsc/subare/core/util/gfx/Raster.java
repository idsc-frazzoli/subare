// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import java.awt.Dimension;
import java.io.Serializable;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.tensor.Scalar;

interface Raster extends Serializable {
  /** @return underlying discrete model */
  DiscreteModel discreteModel();

  /** @return dimension of raster */
  Dimension dimension();

  // TODO functions below are EXPERIMENTAL document
  Scalar scaleLoss();

  Scalar scaleQdelta();

  int joinAlongDimension();

  int magify();
}
