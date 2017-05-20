// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.opt.Interpolation;

/** action value iteration for gambler's dilemma */
class AVI_Gambler {
  private static final Tensor BASE = Tensors.vector(255);

  public static void main(String[] args) throws Exception {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler, RealScalar.ONE);
    avi.untilBelow(RealScalar.of(1e-3));
    {
      Interpolation colorscheme = Colorscheme.classic();
      final Tensor tensor = Array.zeros(101, 51, 4);
      DiscreteQsa qsa = avi.qsa();
      final Scalar max = qsa.getMax();
      for (Tensor state : gambler.states()) {
        for (Tensor action : gambler.actions(state)) {
          Scalar sca = qsa.value(state, action);
          int s = state.Get().number().intValue();
          int a = 50 - action.Get().number().intValue();
          tensor.set(colorscheme.get(BASE.multiply(sca.divide(max))), s, a);
        }
      }
      Export.of(UserHome.file("qsa_gambler.png"), ImageResize.of(tensor, 4));
    }
    DiscreteVs dvs = DiscreteUtils.createVs(gambler, avi.qsa());
    // dvs.print();
    Put.of(UserHome.file("ex403_qsa_values"), dvs.values());
    System.out.println("done.");
  }
}
