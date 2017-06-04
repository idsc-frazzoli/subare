// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.List;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
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
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum GridworldHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    ActionValueIteration avi = new ActionValueIteration(gridworld, gridworld);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }

  private static final Tensor BASE = Tensors.vector(255);
  private static final int MAGNIFY = 7;

  static Tensor render(Gridworld gridworld, DiscreteVs vs) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(gridworld.NX, gridworld.NY, 4);
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    for (Tensor state : gridworld.states()) {
      Scalar sca = scaled.value(state);
      int sx = state.Get(0).number().intValue();
      int sy = state.Get(1).number().intValue();
      tensor.set(colorscheme.get(BASE.multiply(sca)), sx, sy);
    }
    return ImageResize.of(tensor, MAGNIFY);
  }

  static Tensor render(Gridworld gridworld, DiscreteQsa scaled) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros((gridworld.NX + 1) * 4 - 1, gridworld.NY, 4);
    Index indexActions = Index.build(gridworld.actions);
    for (Tensor state : gridworld.states())
      for (Tensor action : gridworld.actions(state)) {
        Scalar sca = scaled.value(state, action);
        int sx = state.Get(0).number().intValue();
        int sy = state.Get(1).number().intValue();
        int a = indexActions.of(action);
        tensor.set(colorscheme.get(BASE.multiply(sca)), sx + (gridworld.NX + 1) * a, sy);
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  public static Tensor joinAll(Gridworld gridworld, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor im1 = render(gridworld, DiscreteQsas.rescaled(qsa));
    Tensor im2 = render(gridworld, DiscreteQsas.logisticDifference(qsa, ref, RealScalar.of(1)));
    List<Integer> list = Dimensions.of(im1);
    list.set(1, MAGNIFY);
    return Join.of(1, im1, Array.zeros(list), im2);
  }
}
