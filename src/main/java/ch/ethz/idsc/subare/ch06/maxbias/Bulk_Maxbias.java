// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import java.awt.Point;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.LearningCompetition;
import ch.ethz.idsc.subare.core.util.LearningContender;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
enum Bulk_Maxbias {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    Maxbias maxbias = new Maxbias(1); // 20, 4/10
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(.5); // 15
    final Scalar losscap = RealScalar.of(.5); // .5
    final Tensor epsilon = Subdivide.of(.2, .01, 100); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, "maxbias_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep, epsilon, errorcap, losscap);
    learningCompetition.nstep = nstep;
    learningCompetition.magnify = 5;
    learningCompetition.period = 100;
    for (Tensor factor : Subdivide.of(.1, 10, 20)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 2, 10)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(maxbias);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(maxbias, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(100, 0.2, 0.01));
        Sarsa sarsa = sarsaType.sarsa(maxbias, DefaultLearningRate.of((Scalar) factor, (Scalar) exponent), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(maxbias, sarsa);
        learningCompetition.put(new Point(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    learningCompetition.doit();
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING, 1);
  }
}
