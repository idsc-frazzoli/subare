// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.List;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
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

enum BlackjackHelper {
  ;
  private static final int MAGNIFY = 5;
  private static final Tensor BASE = Tensors.vector(255);

  public static Tensor render(Blackjack blackjack, DiscreteQsa qsa) {
    return render(blackjack, DiscreteUtils.createVs(blackjack, qsa));
  }

  public static Tensor render(Blackjack blackjack, DiscreteVs vs) {
    Interpolation colorscheme = Colorscheme.classic();
    Tensor tensor = Array.zeros(20 + 2, 10, 4);
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    for (Tensor state : blackjack.states())
      if (state.length() == 3) {
        int useAce = state.Get(0).number().intValue();
        int player = state.Get(1).number().intValue() - 12;
        int dealer = state.Get(2).number().intValue() - 1;
        Scalar max = scaled.value(state); //
        tensor.set(colorscheme.get(BASE.multiply(max)), dealer + 12 * useAce, 9 - player);
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  public static Tensor render(Blackjack blackjack, PolicyInterface policyInterface) {
    Interpolation colorscheme = Colorscheme.classic();
    Tensor tensor = Array.zeros(20 + 2, 10, 4);
    for (Tensor state : blackjack.states())
      if (state.length() == 3) {
        {
          Tensor action = RealScalar.ZERO;
          Scalar sca = policyInterface.policy(state, action);
          int useAce = state.Get(0).number().intValue();
          int player = state.Get(1).number().intValue() - 12;
          int dealer = state.Get(2).number().intValue() - 1;
          tensor.set(colorscheme.get(BASE.multiply(sca)), dealer + 12 * useAce, 9 - player);
        }
      }
    return ImageResize.of(tensor, MAGNIFY);
  }

  public static Tensor joinAll(Blackjack blackjack, DiscreteQsa qsa) {
    Tensor im1 = render(blackjack, DiscreteUtils.createVs(blackjack, qsa));
    PolicyInterface pi = GreedyPolicy.bestEquiprobable(blackjack, qsa);
    Tensor im2 = render(blackjack, pi);
    List<Integer> list = Dimensions.of(im1);
    list.set(1, MAGNIFY * 2);
    return Join.of(1, im1, Array.zeros(list), im2);
  }
}
