// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;

/** reproduces Figure 6.4 on p.139 */
class VI_Windygrid {
  public VI_Windygrid(Windygrid windyGrid) {
    ValueIteration vi = new ValueIteration(windyGrid, RealScalar.ONE);
    vi.untilBelow(DecimalScalar.of(.001));
    final Tensor values = vi.vs().values();
    System.out.println("iterations=" + vi.iterations());
    System.out.println(values);
    PolicyInterface policyInterface = GreedyPolicy.bestEquiprobableGreedy(windyGrid, vi.vs());
    EpisodeInterface mce = windyGrid.kickoff(policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(stepInterface.action() + " -> " + state);
    }
  }

  public static void main(String[] args) {
    new VI_Windygrid(Windygrid.createFour()); // reaches in
    new VI_Windygrid(Windygrid.createKing()); // reaches in 7
  }
}
