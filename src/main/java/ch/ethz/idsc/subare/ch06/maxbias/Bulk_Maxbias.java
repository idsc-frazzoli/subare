// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import java.awt.Point;

import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.LearningCompetition;
import ch.ethz.idsc.subare.core.util.LearningContender;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
class Bulk_Maxbias {
  public static void main(String[] args) throws Exception {
    Maxbias maxbias = new Maxbias(1); // 20, 4/10
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias); // true q-function, for error measurement
    // ---
    SarsaType sarsaType = SarsaType.original;
    final Scalar errorcap = RealScalar.of(.5); // 15
    final Tensor epsilon = Subdivide.of(.2, .01, 100); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, "maxbias_" + sarsaType.name() + "_E" + epsilon.Get(0), epsilon, errorcap);
    learningCompetition.NSTEP = 1;
    learningCompetition.MAGNIFY = 5;
    learningCompetition.PERIOD = 100;
    for (Tensor factor : Subdivide.of(.1, 10, 20)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 2, 10)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(maxbias);
        Sarsa sarsa = sarsaType.supply(maxbias, qsa, DefaultLearningRate.of(factor.Get(), exponent.Get()));
        LearningContender learningContender = LearningContender.sarsa(maxbias, sarsa);
        learningCompetition.put(new Point(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    learningCompetition.doit();
  }
}
