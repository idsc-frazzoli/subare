// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.List;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.ImageResize;
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

enum GamblerHelper {
  ;
  private static final int MAGNIFY = 2;

  static DiscreteQsa getOptimalQsa(Gambler gambler) {
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }

  public static DiscreteVs getOptimalVs(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    return vi.vs();
  }

  public static PolicyInterface getOptimalPolicy(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobable(gambler, vi.vs());
  }

  private static final Tensor BASE = Tensors.vector(255);

  public static Tensor render(Gambler gambler, DiscreteQsa scaled) {
    Interpolation colorscheme = Colorscheme.classic();
    final int length = gambler.states().length();
    final int sizea = (length + 1) / 2;
    final int offset = (length - 1) / 2;
    final Tensor tensor = Array.zeros(length, sizea, 4);
    for (Tensor state : gambler.states())
      for (Tensor action : gambler.actions(state)) {
        Scalar sca = scaled.value(state, action);
        int s = state.Get().number().intValue();
        int a = offset - action.Get().number().intValue();
        tensor.set(colorscheme.get(BASE.multiply(sca)), s, a);
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  public static Tensor render(Gambler gambler, PolicyInterface policyInterface) {
    Interpolation colorscheme = Colorscheme.classic();
    final int length = gambler.states().length();
    final int sizea = (length + 1) / 2;
    final int offset = (length - 1) / 2;
    final Tensor tensor = Array.zeros(length, sizea, 4);
    for (Tensor state : gambler.states())
      for (Tensor action : gambler.actions(state)) {
        Scalar sca = policyInterface.policy(state, action);
        int s = state.Get().number().intValue();
        int a = offset - action.Get().number().intValue();
        tensor.set(colorscheme.get(BASE.multiply(sca)), s, a);
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  public static Tensor joinAll(Gambler gambler, DiscreteQsa qsa) {
    Tensor im1 = render(gambler, TensorValuesUtils.rescaled(qsa));
    PolicyInterface pi = GreedyPolicy.bestEquiprobable(gambler, qsa);
    Tensor im2 = render(gambler, pi);
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 3 * MAGNIFY);
    return Join.of(0, im1, Array.zeros(list), im2);
  }

  public static Tensor joinAll(Gambler gambler, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor im1 = render(gambler, TensorValuesUtils.rescaled(qsa));
    PolicyInterface pi = GreedyPolicy.bestEquiprobable(gambler, qsa);
    Tensor im2 = render(gambler, pi);
    Tensor im3 = render(gambler, TensorValuesUtils.logisticDifference(qsa, ref, RealScalar.of(15)));
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 3 * MAGNIFY);
    return Join.of(0, im1, Array.zeros(list), im2, Array.zeros(list), im3);
  }
}
