// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.StateActionRasters;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum GridworldHelper {
  ;
  // ---
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    return ActionValueIterations.getOptimal(gridworld, DecimalScalar.of(.0001));
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

  static Tensor render(Gridworld gridworld, DiscreteQsa qsa) {
    return ImageResize.of(StateActionRasters.render(new GridworldRaster(gridworld), qsa), MAGNIFY);
  }

  public static Tensor joinAll(Gridworld gridworld, DiscreteQsa qsa, DiscreteQsa ref) {
    return ImageResize.of(StateActionRasters.qsaRef(new GridworldRaster(gridworld), qsa, ref), MAGNIFY);
  }
}
