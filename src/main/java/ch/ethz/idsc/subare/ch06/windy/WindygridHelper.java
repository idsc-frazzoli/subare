// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import java.util.List;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.StateActionRasters;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;

enum WindygridHelper {
  ;
  static DiscreteQsa getOptimalQsa(Windygrid windygrid) {
    return ActionValueIterations.solve(windygrid, DecimalScalar.of(.0001));
  }

  private static final int MAGNIFY = 6;

  static Tensor render(Windygrid windygrid, DiscreteQsa scaled) {
    return ImageResize.of(StateActionRasters.render(new WindygridRaster(windygrid), scaled), MAGNIFY);
  }

  public static Tensor joinAll(Windygrid windygrid, DiscreteQsa qsa, DiscreteQsa ref) {
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    Tensor im1 = StateActionRasters.render(windygridRaster, TensorValuesUtils.rescaled(qsa));
    Tensor im2 = StateActionRasters.render(windygridRaster, TensorValuesUtils.logisticDifference(qsa, ref, RealScalar.ONE));
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 2);
    return ImageResize.of(Join.of(0, im1, Array.zeros(list), im2), MAGNIFY);
  }
}
