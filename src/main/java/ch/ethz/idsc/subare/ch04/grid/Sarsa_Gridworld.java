// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.Put;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum Sarsa_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 10;
    Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_" + sarsaType + "" + nstep + ".gif"), 250)) {
      LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
      Sarsa sarsa = sarsaType.supply(gridworld, learningRate, qsa);
      for (int index = 0; index < batches; ++index) {
        gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        Infoline.print(gridworld, index, ref, qsa);
        Scalar explore = epsilon.Get(index);
        Policy policy = EGreedyPolicy.bestEquiprobable(gridworld, qsa, explore);
        // sarsa.supplyPolicy(() -> policy);
        sarsa.setExplore(epsilon.Get(index));
        ExploringStarts.batch(gridworld, policy, nstep, sarsa);
      }
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa);
    Put.of(UserHome.file("gridworld_" + sarsaType), vs.values());
    Policy policy = GreedyPolicy.bestEquiprobable(gridworld, qsa);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policy);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    int nstep = 2;
    handle(SarsaType.ORIGINAL, nstep);
    handle(SarsaType.EXPECTED, nstep);
    handle(SarsaType.QLEARNING, nstep);
  }
}
