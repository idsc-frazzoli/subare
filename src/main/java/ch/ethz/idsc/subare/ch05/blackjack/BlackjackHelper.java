// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.img.ArrayPlot;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;

enum BlackjackHelper {
  ;
  private static final int MAGNIFY = 5;

  // FIXME magnify irregular
  public static Tensor render(Blackjack blackjack, Policy policy) {
    BlackjackRaster blackjackRaster = new BlackjackRaster(blackjack);
    Dimension dimension = blackjackRaster.dimensionStateRaster();
    Tensor tensor = Array.of(list -> DoubleScalar.INDETERMINATE, dimension.height, dimension.width);
    for (Tensor state : blackjack.states()) {
      Point point = blackjackRaster.point(state);
      if (point != null) {
        Tensor action = RealScalar.ZERO;
        tensor.set(policy.probability(state, action), point.x, point.y);
      }
    }
    return ArrayPlot.of(tensor, ColorDataGradients.CLASSIC);
  }

  public static Tensor render(Blackjack blackjack, DiscreteQsa qsa) {
    return StateRasters.vs_rescale(new BlackjackRaster(blackjack), qsa);
  }

  public static Tensor joinAll(Blackjack blackjack, DiscreteQsa qsa) {
    Tensor im1 = render(blackjack, qsa);
    Policy policy = GreedyPolicy.of(blackjack, qsa);
    Tensor im2 = render(blackjack, policy);
    List<Integer> list = Dimensions.of(im1);
    list.set(1, 2);
    return ImageResize.nearest(Join.of(1, im1, Array.zeros(list), im2), MAGNIFY);
  }
}
