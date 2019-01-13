// code by jph
package ch.ethz.idsc.subare.ch04.grid;

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
enum Bulk_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    Gridworld gambler = new Gridworld(); // 20, 4/10
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gambler); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(20); // 15
    final Scalar losscap = RealScalar.of(5); // .5
    final Tensor epsilon = Subdivide.of(.1, .01, 100); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, "gridworld_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep, epsilon, errorcap, losscap);
    learningCompetition.nstep = nstep;
    learningCompetition.magnify = 5;
    learningCompetition.period = 100;
    for (Tensor factor : Subdivide.of(.1, 10, 10)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 1.3, 10)) { // .51 for qlearning use upper bound == 2, else == 1
        DiscreteQsa qsa = DiscreteQsa.build(gambler);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(100, 0.1, 0.01));
        Sarsa sarsa = sarsaType.sarsa(gambler, DefaultLearningRate.of(factor.Get(), exponent.Get()), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(gambler, sarsa);
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
