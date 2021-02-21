// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;

/** reproduces Figure 6.4 on p.139 */
enum VI_Windygrid {
  ;
  public static void simulate(Windygrid windygrid) {
    ValueIteration vi = new ValueIteration(windygrid, windygrid);
    vi.untilBelow(RealScalar.of(.001));
    final Tensor values = vi.vs().values();
    System.out.println("iterations=" + vi.iterations());
    System.out.println(values);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(windygrid, vi.vs(), null);
    EpisodeInterface episodeInterface = EpisodeKickoff.single(windygrid, policy);
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      System.out.println(stepInterface.prevState() + " + " + stepInterface.action() + " -> " + stepInterface.nextState());
    }
  }

  public static void main(String[] args) {
    simulate(Windygrid.createFour()); // reaches in
    simulate(Windygrid.createKing()); // reaches in 7
  }
}
