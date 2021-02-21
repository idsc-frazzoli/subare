// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.tensor.RealScalar;

/* package */ enum RSTQP_Gridworld {
  ;
  public static void main(String[] args) {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIterations.solve(gridworld, RealScalar.of(0.0001));
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StepDigest stepDigest = Random1StepTabularQPlanning.of( //
        gridworld, qsa, ConstantLearningRate.one());
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.print(gridworld, index, ref, qsa);
      TabularSteps.batch(gridworld, gridworld, stepDigest);
      if (infoline.isLossfree())
        break;
    }
  }
}
