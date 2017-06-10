// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.List;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
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

enum CliffwalkHelper {
  ;
  // ---
  static DiscreteQsa getOptimalQsa(Cliffwalk cliffwalk) {
    ActionValueIteration avi = new ActionValueIteration(cliffwalk, cliffwalk);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }

  static PolicyInterface getOptimalPolicy(Cliffwalk cliffwalk) {
    ValueIteration vi = new ValueIteration(cliffwalk);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobable(cliffwalk, vi.vs());
  }

  private static final Tensor BASE = Tensors.vector(255);
  private static final int MAGNIFY = 6;

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
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(cliffwalk.NX, (cliffwalk.NY + 1) * 4 - 1, 4);
    Index indexActions = Index.build(cliffwalk.actions);
    for (Tensor state : cliffwalk.states())
      for (Tensor action : cliffwalk.actions(state)) {
        Scalar sca = scaled.value(state, action);
        int sx = state.Get(0).number().intValue();
        int sy = state.Get(1).number().intValue();
        int a = indexActions.of(action);
        tensor.set(colorscheme.get(BASE.multiply(sca)), sx, sy + (cliffwalk.NY + 1) * a);
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  static Tensor joinAll(Cliffwalk gambler, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor im1 = render(gambler, TensorValuesUtils.rescaled(qsa));
    Tensor im2 = render(gambler, TensorValuesUtils.logisticDifference(qsa, ref, RealScalar.ONE));
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 2 * MAGNIFY);
    return Join.of(0, im1, Array.zeros(list), im2);
  }
}
