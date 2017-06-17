// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ActionValueStatistics;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

class RSTQP_Maxbias {
  public static void main(String[] args) {
    Maxbias maxbias = new Maxbias(3);
    DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        maxbias, qsa, DefaultLearningRate.of(3, 0.51));
    ActionValueStatistics avs = new ActionValueStatistics(maxbias);
    int EPISODES = 5000;
    for (int index = 0; index < 500; ++index)
      TabularSteps.batch(maxbias, maxbias, rstqp, avs);
    Infoline.print(maxbias, EPISODES, ref, qsa);
    System.out.println("---");
    ActionValueIteration avi = new ActionValueIteration(maxbias, avs);
    avi.untilBelow(RealScalar.of(.0001));
    avi.qsa().print();
    {
      Scalar error = DiscreteValueFunctions.distance(ref, avi.qsa());
      System.out.println("avs error=" + error);
    }
  }
}
