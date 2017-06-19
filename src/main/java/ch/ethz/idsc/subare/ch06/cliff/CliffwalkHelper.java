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
import ch.ethz.idsc.subare.core.util.StateRasters;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;

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

  private static final int MAGNIFY = 5;

  static Tensor render(Cliffwalk cliffwalk, DiscreteVs vs) {
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    return ImageResize.of(StateRasters.render(new CliffwalkStateRaster(cliffwalk), scaled), MAGNIFY);
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
    return ImageResize.of(Join.of(0, image1, Array.zeros(list), image2), cliffwalkRaster.magify());
  }
}
