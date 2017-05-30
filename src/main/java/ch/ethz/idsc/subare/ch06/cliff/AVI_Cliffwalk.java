// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import java.io.IOException;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.io.Export;

/** action value iteration for cliff walk */
class AVI_Cliffwalk {
  public static void main(String[] args) throws IOException {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    ActionValueIteration avi = new ActionValueIteration(cliffwalk, cliffwalk);
    avi.untilBelow(DecimalScalar.of(.0001));
    Export.of(UserHome.file("Pictures/cliffwalk_qsa_avi.png"), CliffwalkHelper.render(cliffwalk, avi.qsa()));
    DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, avi.qsa());
    vs.print();
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(cliffwalk, avi.qsa());
    greedyPolicy.print(cliffwalk.states());
  }
}
