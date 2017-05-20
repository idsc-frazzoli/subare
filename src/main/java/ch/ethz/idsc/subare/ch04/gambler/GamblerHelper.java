package ch.ethz.idsc.subare.ch04.gambler;

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
import ch.ethz.idsc.tensor.opt.Interpolation;

enum GamblerHelper {
  ;
  private static final Tensor BASE = Tensors.vector(255);

  public static PolicyInterface getOptimalPolicy(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler, RealScalar.ONE);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobableGreedy(gambler, vi.vs());
  }

  public static Tensor render(Gambler gambler, DiscreteQsa qsa) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(101, 51, 4); // TODO lookup constants
    final Scalar max = qsa.getMax();
    for (Tensor state : gambler.states())
      for (Tensor action : gambler.actions(state)) {
        Scalar sca = qsa.value(state, action);
        int s = state.Get().number().intValue();
        int a = 50 - action.Get().number().intValue();
        tensor.set(colorscheme.get(BASE.multiply(sca.divide(max))), s, a);
      }
    return ImageResize.of(tensor, 4);
  }
}
