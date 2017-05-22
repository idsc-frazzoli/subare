// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Export;

/** */
class VI_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    ValueIteration vi = new ValueIteration(cliffwalk, RealScalar.ONE);
    vi.untilBelow(DecimalScalar.of(.0001));
    Export.of(UserHome.file("Pictures/cliffwalk_qsa_vi.png"), CliffwalkHelper.render(cliffwalk, vi.vs()));
    // GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(cliffWalk, values);
    // greedyPolicy.print(cliffWalk.states());
    // Index statesIndex = Index.build(cliffWalk.states());
    // for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
    // Tensor state = statesIndex.get(stateI);
    // System.out.println(state + " " + values.get(stateI).map(ROUND));
    // }
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobableGreedy(cliffwalk, vi.vs());
    EpisodeInterface mce = cliffwalk.kickoff(policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
