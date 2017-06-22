// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.Colorscheme;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum BlackjackHelper {
  ;
  private static final int MAGNIFY = 5;
  private static final Interpolation COLORSCHEME = Colorscheme.classic();
  private static final Tensor BASE = Tensors.vector(255);

  // FIXME magnify irregular
  public static Tensor render(Blackjack blackjack, Policy policy) {
    BlackjackRaster blackjackRaster = new BlackjackRaster(blackjack);
    Dimension dimension = blackjackRaster.dimensionStateRaster();
    Tensor tensor = Array.zeros(dimension.width, dimension.height, 4);
    for (Tensor state : blackjack.states()) {
      Point point = blackjackRaster.point(state);
      if (point != null) {
        Tensor action = RealScalar.ZERO;
        Scalar sca = policy.probability(state, action);
        tensor.set(COLORSCHEME.get(BASE.multiply(sca)), point.x, point.y);
      }
    }
    return tensor;
  }

  public static Tensor render(Blackjack blackjack, DiscreteQsa qsa) {
    return StateRasters.vs_rescale(new BlackjackRaster(blackjack), qsa);
  }

  public static Tensor joinAll(Blackjack blackjack, DiscreteQsa qsa) {
    Tensor im1 = render(blackjack, qsa);
    Policy policy = GreedyPolicy.bestEquiprobable(blackjack, qsa);
    Tensor im2 = render(blackjack, policy);
    List<Integer> list = Dimensions.of(im1);
    list.set(1, 2);
    return ImageResize.of(Join.of(1, im1, Array.zeros(list), im2), MAGNIFY);
  }
}
