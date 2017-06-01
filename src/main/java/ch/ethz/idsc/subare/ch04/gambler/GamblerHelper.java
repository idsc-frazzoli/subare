// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.List;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum GamblerHelper {
  ;
  public static PolicyInterface getOptimalPolicy(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobableGreedy(gambler, vi.vs());
  }

  private static final Tensor BASE = Tensors.vector(255);

  public static Tensor render(Gambler gambler, DiscreteQsa qsa) {
    Interpolation colorscheme = Colorscheme.classic();
    final int length = gambler.states().length();
    final int sizea = (length + 1) / 2;
    final int offset = (length - 1) / 2;
    final Tensor tensor = Array.zeros(length, sizea, 4);
    DiscreteQsa scaled = qsa.create(Rescale.of(qsa.values()).flatten(0));
    for (Tensor state : gambler.states())
      for (Tensor action : gambler.actions(state)) {
        Scalar sca = scaled.value(state, action);
        int s = state.Get().number().intValue();
        int a = offset - action.Get().number().intValue();
        tensor.set(colorscheme.get(BASE.multiply(sca)), s, a);
      }
    return ImageResize.of(tensor, 2);
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
    return ImageResize.of(tensor, 2);
  }

  public static Tensor joinAll(Gambler gambler, DiscreteQsa qsa) {
    Tensor im1 = render(gambler, qsa);
    PolicyInterface pi = GreedyPolicy.bestEquiprobableGreedy(gambler, qsa);
    Tensor im2 = render(gambler, pi);
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 6 * 2);
    return Join.of(0, im1, Array.zeros(list), im2);
  }
}
