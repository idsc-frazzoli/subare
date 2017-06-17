// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.List;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.StateActionRasters;
import ch.ethz.idsc.subare.util.Colorscheme;
import ch.ethz.idsc.subare.util.ImageResize;
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

enum CliffwalkHelper {
  ;
  static DiscreteQsa getOptimalQsa(Cliffwalk cliffwalk) {
    return ActionValueIterations.solve(cliffwalk, DecimalScalar.of(.0001));
  }

  static Policy getOptimalPolicy(Cliffwalk cliffwalk) {
    ValueIteration vi = new ValueIteration(cliffwalk, cliffwalk);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobable(cliffwalk, vi.vs());
  }

  private static final Tensor BASE = Tensors.vector(255);
  private static final int MAGNIFY = 6;

  // TODO implement state raster
  static Tensor render(Cliffwalk cliffwalk, DiscreteVs vs) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(cliffwalk.NX, cliffwalk.NY, 4);
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    for (Tensor state : cliffwalk.states()) {
      Scalar sca = scaled.value(state);
      int sx = state.Get(0).number().intValue();
      int sy = state.Get(1).number().intValue();
      tensor.set(colorscheme.get(BASE.multiply(sca)), sx, sy);
    }
    return ImageResize.of(tensor, MAGNIFY);
  }

  static Tensor render(Cliffwalk cliffwalk, DiscreteQsa scaled) {
    return ImageResize.of(StateActionRasters.render(new CliffwalkRaster(cliffwalk), scaled), MAGNIFY);
  }

  static Tensor joinAll(Cliffwalk cliffwalk, DiscreteQsa qsa, DiscreteQsa ref) {
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    Tensor image1 = StateActionRasters.render(cliffwalkRaster, DiscreteValueFunctions.rescaled(qsa));
    Tensor image2 = StateActionRasters.render(cliffwalkRaster, DiscreteValueFunctions.logisticDifference(qsa, ref));
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 2);
    return ImageResize.of(Join.of(0, image1, Array.zeros(list), image2), MAGNIFY);
  }
}
