// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Put;

/** action value iteration for gambler's dilemma */
class AVI_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler, RealScalar.ONE);
    avi.untilBelow(RealScalar.of(1e-3));
    Export.of(UserHome.file("gambler_qsa_avi.png"), GamblerHelper.render(gambler, avi.qsa()));
    DiscreteVs dvs = DiscreteUtils.createVs(gambler, avi.qsa());
    // dvs.print();
    Put.of(UserHome.file("ex403_qsa_values"), dvs.values());
    System.out.println("done.");
  }
}
