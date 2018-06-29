// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.Put;

/** Double Sarsa for gridworld */
enum Double_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    System.out.println("double " + sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 40;
    Tensor epsilon = Subdivide.of(.1, .01, batches); // used in egreedy
    DiscreteQsa qsa1 = DiscreteQsa.build(gridworld);
    DiscreteQsa qsa2 = DiscreteQsa.build(gridworld);
    DoubleSarsa doubleSarsa = new DoubleSarsa(sarsaType, gridworld, //
        qsa1, qsa2, //
        DefaultLearningRate.of(5, .51), //
        DefaultLearningRate.of(5, .51));
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_double_" + sarsaType + "" + nstep + ".gif"), 150)) {
      for (int index = 0; index < batches; ++index) {
        if (batches - 10 < index)
          Infoline.print(gridworld, index, ref, qsa1);
        doubleSarsa.setExplore(epsilon.Get(index));
        ExploringStarts.batch(gridworld, doubleSarsa.getEGreedy(), nstep, doubleSarsa);
        gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa1, ref));
      }
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa1);
    Put.of(UserHome.file("gridworld_" + sarsaType), vs.values());
    Policy policy = GreedyPolicy.bestEquiprobable(gridworld, qsa1);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policy);
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
