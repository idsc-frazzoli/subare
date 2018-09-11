// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Tensor;

/** reproduces Figure 6.4 on p.139 */
enum VI_Windygrid {
  ;
  public static void simulate(Windygrid windygrid) {
    ValueIteration vi = new ValueIteration(windygrid, windygrid);
    vi.untilBelow(DecimalScalar.of(.001));
    final Tensor values = vi.vs().values();
    System.out.println("iterations=" + vi.iterations());
    System.out.println(values);
    Policy policy = GreedyPolicy.of(windygrid, vi.vs());
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
