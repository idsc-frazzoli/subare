// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

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
enum Bulk_Wireloop {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    String name = "wire4";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x); // 20, 4/10
    final DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(15); // 15
    final Scalar losscap = RealScalar.of(.05); // .5
    final Tensor epsilon = Subdivide.of(.2, .05, 40); // .2, .6
    int x = 0;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, name + "_Q_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep, //
        epsilon, errorcap, losscap);
    learningCompetition.nstep = 1;
    learningCompetition.magnify = 5;
    for (Tensor factor : Subdivide.of(.1, 10, 20)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 1.5, 20)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(wireloop);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(40, 0.2, 0.05));
        Sarsa sarsa = sarsaType.supply(wireloop, DefaultLearningRate.of(factor.Get(), exponent.Get()), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(wireloop, sarsa);
        learningCompetition.put(new Point(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    learningCompetition.doit();
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING, 2);
  }
}
