// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import java.util.List;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum WindygridHelper {
  ;
  // ---
  static DiscreteQsa getOptimalQsa(Windygrid windygrid) {
    ActionValueIteration avi = new ActionValueIteration(windygrid, windygrid);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }

  private static final Tensor BASE = Tensors.vector(255);
  private static final int MAGNIFY = 6;

  static Tensor render(Windygrid windygrid, DiscreteQsa scaled) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(Windygrid.NX, (Windygrid.NY + 1) * 4, 4);
    Index indexActions = Index.build(windygrid.actions);
    for (Tensor state : windygrid.states())
      for (Tensor action : windygrid.actions(state)) {
        Scalar sca = scaled.value(state, action);
        int sx = state.Get(0).number().intValue();
        int sy = state.Get(1).number().intValue();
        int a = indexActions.of(action);
        tensor.set(colorscheme.get(BASE.multiply(sca)), sx, sy + (Windygrid.NY + 1) * a);
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  public static Tensor joinAll(Windygrid gambler, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor im1 = render(gambler, DiscreteQsas.rescaled(qsa));
    Tensor im2 = render(gambler, DiscreteQsas.logisticDifference(qsa, ref, RealScalar.ONE));
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 2 * MAGNIFY);
    return Join.of(0, im1, Array.zeros(list), im2);
  }
}
