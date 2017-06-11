// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Export;

/** value iteration for cliffwalk */
class VI_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    ValueIteration vi = new ValueIteration(cliffwalk, cliffwalk);
    vi.untilBelow(DecimalScalar.of(.0001));
    DiscreteVs vs = vi.vs();
    DiscreteVs vr = DiscreteUtils.createVs(cliffwalk, ref);
    Scalar error = TensorValuesUtils.distance(vs, vr);
    System.out.println("error=" + error);
    Export.of(UserHome.file("Pictures/cliffwalk_qsa_vi.png"), CliffwalkHelper.render(cliffwalk, vi.vs()));
    // GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(cliffWalk, values);
    // greedyPolicy.print(cliffWalk.states());
    // Index statesIndex = Index.build(cliffWalk.states());
    // for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
    // Tensor state = statesIndex.get(stateI);
    // System.out.println(state + " " + values.get(stateI).map(ROUND));
    // }
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobable(cliffwalk, vi.vs());
    EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
