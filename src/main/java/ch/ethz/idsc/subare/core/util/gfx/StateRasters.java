// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.img.ArrayPlot;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.sca.Clip;

public enum StateRasters {
  ;
  public static Point canonicPoint(Tensor state) {
    return new Point( //
        state.Get(0).number().intValue(), //
        state.Get(1).number().intValue());
  }

  /** @param stateActionRaster
   * @param vs scaled to contain values in the interval [0, 1]
   * @return */
  private static Tensor _render(StateRaster stateRaster, DiscreteVs vs) {
    DiscreteModel discreteModel = stateRaster.discreteModel();
    Dimension dimension = stateRaster.dimensionStateRaster();
    Tensor tensor = Array.zeros(dimension.width, dimension.height).map(scalar -> DoubleScalar.INDETERMINATE);
    for (Tensor state : discreteModel.states()) {
      Point point = stateRaster.point(state);
      if (point != null)
        tensor.set(vs.value(state), point.x, point.y);
    }
    return ArrayPlot.of(tensor, ColorDataGradients.CLASSIC);
  }

  private static Tensor _vs(StateRaster stateRaster, DiscreteQsa qsa) {
    return _render(stateRaster, DiscreteUtils.createVs(stateRaster.discreteModel(), qsa));
  }

  private static Tensor _vs_rescale(StateRaster stateRaster, DiscreteQsa qsa) {
    DiscreteVs vs = DiscreteUtils.createVs(stateRaster.discreteModel(), qsa);
    return _render(stateRaster, vs.create(Rescale.of(vs.values()).flatten(0)));
  }

  /***************************************************/
  public static Tensor vs(StateRaster stateRaster, DiscreteVs vs) {
    return ImageResize.nearest(_render(stateRaster, vs), stateRaster.magnify());
  }

  public static Tensor vs_rescale(StateRaster stateRaster, DiscreteVs vs) {
    return vs(stateRaster, vs.create(Rescale.of(vs.values()).flatten(0)));
  }

  public static Tensor vs(StateRaster stateRaster, DiscreteQsa qsa) {
    return vs(stateRaster, DiscreteUtils.createVs(stateRaster.discreteModel(), qsa));
  }

  public static Tensor vs_rescale(StateRaster stateRaster, DiscreteQsa qsa) {
    DiscreteVs vs = DiscreteUtils.createVs(stateRaster.discreteModel(), qsa);
    return vs(stateRaster, vs.create(Rescale.of(vs.values()).flatten(0)));
  }

  public static Tensor qsaLossRef(StateRaster stateRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = _vs_rescale(stateRaster, DiscreteValueFunctions.rescaled(qsa));
    DiscreteVs loss = Loss.perState(stateRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().flatten(0) //
        .map(tensor -> tensor.multiply(stateRaster.scaleLoss())) //
        .map(Clip.UNIT::of));
    Tensor image2 = _render(stateRaster, loss);
    Tensor image3 = _vs(stateRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, stateRaster.scaleQdelta()));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateRaster.joinAlongDimension();
    list.set(dim, 1);
    return ImageResize.nearest( //
        Join.of(dim, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateRaster.magnify());
  }
}
