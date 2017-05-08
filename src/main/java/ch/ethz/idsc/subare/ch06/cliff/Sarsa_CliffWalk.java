// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** sarsa applied to cliff walk */
class Sarsa_CliffWalk {
  public static void main(String[] args) {
    CliffWalk windyGrid = new CliffWalk();
    PolicyInterface policy = new EquiprobablePolicy(windyGrid);
    DiscreteQsa qsa = DiscreteQsa.build(windyGrid);
    System.out.println(qsa.size());
    for (int c = 0; c < 100; ++c) {
      System.out.println(c);
      Sarsa sarsa = new Sarsa( //
          windyGrid, policy, //
          windyGrid, //
          qsa, RealScalar.ONE, RealScalar.of(.25));
      sarsa.simulate(3);
      policy = EGreedyPolicy.bestEquiprobable(windyGrid, qsa, RealScalar.of(.1));
      // policy = GreedyPolicy.bestEquiprobableGreedy(randomWalk, qsa); //
    }
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    EpisodeInterface mce = windyGrid.kickoff(policy);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }
}
