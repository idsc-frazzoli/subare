// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Put;

/** Double Sarsa for gridworld */
enum Double_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    System.out.println("double " + sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 40;
    DiscreteQsa qsa1 = DiscreteQsa.build(gridworld);
    DiscreteQsa qsa2 = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, DiscreteQsa.build(gridworld), sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    EGreedyPolicy policy1 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa1, sac1);
    policy1.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    EGreedyPolicy policy2 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa2, sac2);
    policy2.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    DoubleSarsa doubleSarsa = sarsaType.doubleSarsa( //
        gridworld, //
        DefaultLearningRate.of(5, .51), //
        qsa1, qsa2, sac1, sac2, policy1, policy2);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gridworld_double_" + sarsaType + "" + nstep + ".gif"), 150, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < batches; ++index) {
        if (batches - 10 < index)
          Infoline.print(gridworld, index, ref, qsa1);
        policy.setQsa(doubleSarsa.qsa());
        policy.setSac(sac);
        ExploringStarts.batch(gridworld, policy, nstep, doubleSarsa);
        animationWriter.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa1, ref));
      }
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, doubleSarsa.qsa());
    Put.of(HomeDirectory.file("gridworld_" + sarsaType), vs.values());
    Policy policyVs = PolicyType.GREEDY.bestEquiprobable(gridworld, vs, null);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyVs);
    while (ei.hasNext()) {
      StepInterface stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.ORIGINAL, 1);
    handle(SarsaType.EXPECTED, 1);
    handle(SarsaType.QLEARNING, 1);
  }
}
